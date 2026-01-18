import contextlib
import copy
import functools
import inspect
import os
import re
import warnings
from typing import Any, Callable, Dict, List, Optional, Set, TypeVar
from typing_extensions import ParamSpec
import torch
from torch._jit_internal import (
from torch.autograd import function
from torch.jit._script import _CachedForward, script, ScriptModule
from torch.jit._state import _enabled, _python_cu
from torch.nn import Module
from torch.testing._comparison import default_tolerances
def graph_diagnostic_info():
    mod_canonicalized = torch._C._jit_pass_canonicalize(traced_func.graph)
    torch._C._jit_pass_inline(mod_canonicalized)
    torch._C._jit_pass_erase_shape_information(mod_canonicalized)
    mod_str = str(mod_canonicalized)
    mod_str = re.sub('___torch_mangle_[0-9]+\\.', '', mod_str)
    check_canonicalized = torch._C._jit_pass_canonicalize(check_mod_func.graph)
    torch._C._jit_pass_inline(check_canonicalized)
    torch._C._jit_pass_erase_shape_information(check_canonicalized)
    check_str = str(check_canonicalized)
    check_str = re.sub('___torch_mangle_[0-9]+\\.', '', check_str)
    graph_diff_errors = None
    if mod_str != check_str:
        import difflib
        graph_diff = difflib.ndiff(mod_str.splitlines(True), check_str.splitlines(True))
        graph_diff_errors = 'Graph diff:\n' + indent(''.join(graph_diff)) + '\n'
        for n_mod, n_check in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
            if str(n_mod) != str(n_check):
                graph_diff_errors += 'First diverging operator:\n'
                node_diff = difflib.ndiff(str(n_mod).splitlines(True), str(n_check).splitlines(True))
                source_printout = 'Node diff:\n' + indent(''.join(node_diff)) + '\n'
                mod_stack = n_mod.sourceRange()
                if mod_stack:
                    source_printout += 'Trace source location:\n' + indent(mod_stack) + '\n'
                check_stack = n_check.sourceRange()
                if check_stack:
                    source_printout += 'Check source location:\n' + indent(check_stack) + '\n'
                graph_diff_errors += source_printout
                break
    tensor_compare_errors = None
    for n_mod, n_check in zip(mod_canonicalized.nodes(), check_canonicalized.nodes()):
        if n_mod.kind() != n_check.kind():
            break
        if n_mod.kind() == 'prim::Constant' and (not (n_mod.mustBeNone() or n_check.mustBeNone())):
            if not n_mod.hasAttribute('value'):
                continue
            if n_mod.kindOf('value') != 't' or n_check.kindOf('value') != 't':
                continue
            mod_tensor_val = n_mod.t('value')
            check_tensor_val = n_check.t('value')
            try:
                torch.testing.assert_close(mod_tensor_val, check_tensor_val, equal_nan=True)
            except (RuntimeError, AssertionError) as e:
                if tensor_compare_errors is None:
                    tensor_compare_errors = ''
                tensor_compare_errors += 'Node:\n' + indent(str(n_mod)) + '\n'
                compare_stack = n_mod.sourceRange()
                if compare_stack:
                    tensor_compare_errors += 'Source Location:\n' + indent(compare_stack) + '\n'
                tensor_compare_errors += 'Comparison exception: ' + indent(str(e))
                break
    return (graph_diff_errors, tensor_compare_errors)