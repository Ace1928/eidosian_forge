import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
def inspect_module(module, target=None, alias=None):
    """Inspect a module object and yielding results from `inspect_function()`
    for each function object in the module.
    """
    alias = {} if alias is None else alias
    for name in dir(module):
        if name.startswith('_'):
            continue
        obj = getattr(module, name)
        supported_types = (pytypes.FunctionType, pytypes.BuiltinFunctionType)
        if not isinstance(obj, supported_types):
            continue
        info = dict(module=module, name=name, obj=obj)
        if obj in alias:
            info['alias'] = alias[obj]
        else:
            alias[obj] = '{module}.{name}'.format(module=module.__name__, name=name)
        info.update(inspect_function(obj, target=target))
        yield info