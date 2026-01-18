import builtins
import collections
import functools
import inspect
import itertools
import logging
import math
import operator
import re
import sys
import threading
import traceback
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any, cast, Callable, Dict, List, Optional, Sequence, Set, Tuple, Type, Union, Iterable
import torch
import torch.fx
import torch.fx.traceback as fx_traceback
from torch.fx.experimental import _config as config
from torch.fx.experimental.recording import (
from torch.fx.experimental.sym_node import SymNode, SymTypes
from torch import SymBool, SymFloat, SymInt
from torch._guards import ShapeGuard, Source, TracingContext
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._sympy.functions import FloorDiv, Mod, IsNonOverlappingAndDenseIndicator
from torch.utils._sympy.solve import try_solve
from torch.utils._sympy.value_ranges import bound_sympy, SymPyValueRangeAnalysis, ValueRanges, ValueRangeError
from torch.utils._sympy.singleton_int import SingletonInt
from torch.utils._traceback import format_frame, CapturedTraceback
from torch._utils_internal import signpost_event
from torch._logging import LazyString
import sympy
from sympy.printing.str import StrPrinter
from sympy.printing.precedence import precedence, PRECEDENCE
def produce_guards(self, placeholders, sources, source_ref=lambda n: n.name(), *, constraint_inputs: Optional[InputList[Union[DimConstraint, Optional[DimList[DimConstraint]]]]]=None, equalities_inputs: Optional[Set[Tuple[Source, Source]]]=None, _simplified=False, ignore_static=True) -> List[str]:
    self.log.info('produce_guards')
    if self.check_recorded_events:
        shape_env = replay_shape_env_events(self.events)
        self.check_equal(shape_env)
    assert len(placeholders) == len(sources)
    Tensorlike = (torch.Tensor, FakeTensorMeta)
    if constraint_inputs is None:
        constraint_inputs = [[None] * t.dim() if isinstance(t, Tensorlike) else None for t in placeholders]
    else:
        assert len(constraint_inputs) == len(placeholders)
        for i, (t, constraint) in enumerate(zip(placeholders, constraint_inputs)):
            if isinstance(t, Tensorlike):
                if constraint is None:
                    constraint_inputs[i] = [None] * t.dim()
                else:
                    assert len(constraint) == t.dim()
            else:
                assert isinstance(t, (SymInt, int))
                assert not isinstance(constraint, list)
    from torch._dynamo.source import TensorPropertySource, TensorProperty, NegateSource
    input_guards = []
    symbol_to_source = collections.defaultdict(list)
    symbol_to_constraints = collections.defaultdict(set)
    constraint_violations: List[Tuple[bool, Callable[[], str]]] = []

    def record_constraint_violation(warn_only, debug_name, msg, hint=None):
        constraint_violations.append((warn_only, debug_name, lambda: f'{msg}{hint()}' if hint else msg))

    def is_dim(src):
        return isinstance(src, TensorPropertySource) and src.prop is TensorProperty.SIZE
    if equalities_inputs:
        source_index = {}
        for i, src in enumerate(sources):
            source_index[src.name()] = i

        def get_symbol(tensor_dim_src):
            fake = placeholders[source_index[tensor_dim_src.base.name()]]
            symint = fake.shape[tensor_dim_src.idx]
            assert isinstance(symint, torch.SymInt)
            return symint.node.expr
        for src1, src2 in equalities_inputs.source_pairs:
            s1, s2 = (get_symbol(src1), get_symbol(src2))
            concrete_val = self.evaluate_expr(sympy.Eq(s1, s2))
            if not concrete_val:
                raise ConstraintViolationError(f'{src1.name()} = {self.var_to_val[s1]} is not equal to {src2.name()} = {self.var_to_val[s2]}')

    def track_symint(source, val, constraint=None):
        log.debug('track_symint %s %s %s', LazyString(source.name), val, constraint)
        assert not isinstance(val, SymInt) or is_symbolic(val)
        if isinstance(val, SymInt) and val.node.maybe_as_int() is not None:
            val = val.node.maybe_as_int()
        if isinstance(val, SymInt):
            s = val.node.expr
            if isinstance(s, sympy.Symbol):
                symbol_to_source[s].append(source)
                if constraint is not None:
                    symbol_to_constraints[s].add(constraint)
            elif isinstance(-s, sympy.Symbol):
                symbol_to_source[-s].append(NegateSource(source))
            else:
                constraint_violated = False
                if isinstance(constraint, StrictMinMaxConstraint):
                    sym_vrs = {x: self.var_to_range.get(x, None) for x in s.free_symbols}
                    if all((vr is not None for vr in sym_vrs.values())):
                        expr_vr = bound_sympy(s, sym_vrs)
                        if expr_vr != constraint.vr:
                            constraint_violated = True
                    else:
                        constraint_violated = True
                elif isinstance(constraint, RelaxedUnspecConstraint):
                    if s.is_number:
                        i = int(s)
                        if i not in (0, 1):
                            constraint_violated = True
                    else:
                        constraint_violated = True
                if constraint_violated:

                    def hint(s):
                        sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(s)
                        return f'{sexpr}.'
                    var_with_range = self.render_range_for_constraint_violation(source, constraint)
                    msg = f'Not all values of {var_with_range} are valid because {self.debug_name(source)} was inferred to be equal to '
                    record_constraint_violation(constraint.warn_only, self.debug_name(source), msg, hint=functools.partial(hint, s))
            input_guards.append((source, s))
        else:
            s = sympy.Integer(val)
            input_guards.append((source, s))
            constraint_violated = False
            if isinstance(constraint, StrictMinMaxConstraint):
                constraint_violated = True
            elif isinstance(constraint, RelaxedUnspecConstraint):
                if val not in (0, 1):
                    constraint_violated = True
            if constraint_violated:
                var_with_range = self.render_range_for_constraint_violation(source, constraint)
                msg = f'Not all values of {var_with_range} are valid because {self.debug_name(source)} was inferred to be a constant ({val}).'
                record_constraint_violation(constraint.warn_only, self.debug_name(source), msg)
    for t, source, constraint in zip(placeholders, sources, constraint_inputs):
        if isinstance(source, str):
            from torch._dynamo.source import LocalSource
            source = LocalSource(source)
        assert isinstance(source, Source)
        if t is None:
            continue
        if isinstance(t, (SymInt, int)):
            track_symint(source, t)
            continue
        assert isinstance(t, Tensorlike)
        sources_and_tensors = [(source, t)]
        if is_traceable_wrapper_subclass(t):
            attrs, _ = t.__tensor_flatten__()
            from torch._dynamo.source import AttrSource
            inner_sources_and_tensors = [(AttrSource(source, attr), getattr(t, attr)) for attr in attrs]
            if t.is_nested:
                sources_and_tensors.extend(inner_sources_and_tensors)
            else:
                sources_and_tensors = inner_sources_and_tensors
        for src, curr_t in sources_and_tensors:
            for i, ss in enumerate(curr_t.size()):
                property_source = TensorPropertySource(src, TensorProperty.SIZE, i)
                track_symint(property_source, ss, constraint[i])
            if not t.is_nested:
                for i, ss in enumerate(curr_t.stride()):
                    track_symint(TensorPropertySource(src, TensorProperty.STRIDE, i), ss)
                track_symint(TensorPropertySource(src, TensorProperty.STORAGE_OFFSET), curr_t.storage_offset())
    exprs = []
    self.dim_constraints = DimConstraints(symbol_to_source, self.var_to_val, set(symbol_to_constraints.keys()), self.source_name_to_debug_name)
    if not _simplified:
        for source, expr in input_guards:
            if self._translation_validation_enabled:
                srcname = source.name()
                if srcname in self.source_to_symbol:
                    self._add_target_expr(sympy.Eq(self.source_to_symbol[srcname], expr))
            if isinstance(expr, sympy.Symbol) and symbol_to_source.get(expr) and (source == symbol_to_source[expr][0]):
                continue
            if ignore_static and isinstance(source, TensorPropertySource):
                if expr.is_number:
                    self.log.debug('Skipping guard %s', f'{source_ref(source)} == {expr}')
                    continue
            if is_dim(source):
                self.dim_constraints.add_equality(source, expr)
            sexpr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
            exprs.append(f'{source_ref(source)} == {sexpr}')
            if isinstance(expr, sympy.Symbol) and expr in symbol_to_constraints and isinstance(source, TensorPropertySource) and (source.prop is TensorProperty.SIZE) and equalities_inputs and (not equalities_inputs.is_equal(source, symbol_to_source[expr][0])):
                msg = f'The values of {self.debug_name(source)} = {source.name()} and {self.debug_name(symbol_to_source[expr][0])} = {symbol_to_source[expr][0].name()} must always be equal.'
                record_constraint_violation(equalities_inputs.warn_only, self.debug_name(source), msg)
    issued = set()

    def issue_guard(guard: ShapeGuard) -> None:
        expr = self.simplify(guard.expr)
        if expr in issued:
            return
        issued.add(expr)
        try:
            is_trivial = False
            if any((is_dim(source) for s in expr.free_symbols for source in symbol_to_source[s])):
                is_trivial = self.dim_constraints.add(expr)
            guard_expr = ShapeGuardPrinter(symbol_to_source, source_ref, self.var_to_sources).doprint(expr)
            exprs.append(guard_expr)
            self._add_target_expr(expr)
            if not is_trivial and len(expr.free_symbols) == 1:
                symbol = next(iter(expr.free_symbols))
                source = symbol_to_source[symbol][0]
                constraints = symbol_to_constraints[symbol]
                for c in constraints:
                    if isinstance(c, StrictMinMaxConstraint):
                        var_with_range = self.render_range_for_constraint_violation(source, c)
                        msg = f'Not all values of {var_with_range} satisfy the generated guard {guard_expr}.'
                        record_constraint_violation(c.warn_only, self.debug_name(source), msg)
                    elif isinstance(c, RelaxedUnspecConstraint):
                        pass
                    else:
                        raise AssertionError(f'unrecognized constraint {c}')
        except Exception:
            self.log.warning('Failing guard allocated at: \n%s', ''.join(guard.stack.format()))
            raise
    for guard in self.guards:
        if self._maybe_evaluate_static(guard.expr) is not None:
            continue
        issue_guard(guard)
    for symbol, guards in self.var_to_guards.items():
        if symbol not in symbol_to_source:
            continue
        for guard in guards:
            if guard is not None:
                issue_guard(guard)
    if not _simplified:
        for symbol, sources in symbol_to_source.items():
            r = self.runtime_var_to_range.get(symbol)
            if r is None:
                if symbol not in self.var_to_range:
                    continue
                r = self.var_to_range[symbol]
            assert sources
            assert symbol.is_integer
            g_lower, g_upper = self.var_to_guards.get(symbol, (None, None))
            bounds = []
            if r.lower != -sympy.oo and g_lower is None:
                if any((is_dim(source) for source in sources)):
                    self.dim_constraints.add(sympy.Ge(symbol, r.lower))
                bounds.append(str(r.lower))
            bounds.append(source_ref(sources[0]))
            if r.upper != sympy.oo and r.upper < sys.maxsize - 1 and (g_upper is None):
                if any((is_dim(source) for source in sources)):
                    self.dim_constraints.add(sympy.Le(symbol, r.upper))
                bounds.append(str(r.upper))
            if len(bounds) > 1:
                exprs.append(' <= '.join(bounds))
    if constraint_violations:
        warn_msgs = []
        error_msgs = []
        debug_names = set()
        for warn_only, debug_name, msg in constraint_violations:
            if warn_only:
                msg = f'  {len(warn_msgs) + 1}. {msg()}'
                warn_msgs.append(msg)
            else:
                msg = f'  - {msg()}'
                error_msgs.append(msg)
                debug_names.add(debug_name)
        if len(error_msgs) > 0:
            debug_names = ', '.join(debug_names)
            err = '\n'.join(error_msgs)
            raise ConstraintViolationError(f'Constraints violated ({debug_names})! For more information, run with TORCH_LOGS=dynamic.\n{err}')
        elif len(warn_msgs) > 0:
            log.debug('%s Warning only constraints violated', len(warn_msgs))
    signpost_event('dynamic', 'produce_guards', {**self.co_fields, **self.counter, 'num_guards': len(exprs), 'free_symbols': sum((1 for v in symbol_to_source.values() if v))})
    if self._translation_validation_enabled:
        from torch.fx.experimental.validator import PopulateValidator
        for ras in self.deferred_runtime_asserts.values():
            for ra in ras:
                self._add_target_expr(ra.expr)
        for sym, vr in self.var_to_range.items():
            if vr.lower != -sympy.oo:
                self._add_target_expr(sympy.Le(vr.lower, sym))
            if vr.upper != sympy.oo:
                self._add_target_expr(sympy.Le(sym, vr.upper))
        with fx_traceback.preserve_node_meta():
            PopulateValidator(self.graph, self.validator).run()
    self._check_translation_validate()
    return exprs