import functools
import inspect
import itertools
import types
from typing import Dict, List
import torch
from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker
def bind_args(self, parent, args, kwargs):
    from .misc import InlinedClosureVariable
    code = self.get_code()
    func = types.FunctionType(code, self.f_globals, self.fn_name.as_python_constant(), tuple(self.defaults.items) if self.defaults else None, tuple((make_cell(None) for _ in range(len(self.get_code().co_freevars)))))
    if self.kwdefaults:
        func.__kwdefaults__ = self.kwdefaults.items
    bound = inspect.signature(func).bind(*args, **kwargs)
    bound.apply_defaults()
    result = dict(bound.arguments.items())
    wrap_args_kwargs(parent.output.root_tx, result)
    closure_cells = init_cellvars(parent, result, code)
    for idx, name in enumerate(code.co_freevars):
        cell = self.closure.items[idx]
        assert getattr(cell, name, name) == name
        assert name not in result
        if isinstance(cell, InlinedClosureVariable):
            cand = parent
            while cand and name not in cand.symbolic_locals:
                cand = cand.parent
            if cand is None:
                raise RuntimeError(f"Couldn't find {name} in the symbolic_locals of the inline interpreter stack")
            result[name] = cand.symbolic_locals[name]
        else:
            closure_cells[name] = self.closure.items[idx]
    return (result, closure_cells)