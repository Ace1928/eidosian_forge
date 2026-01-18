import functools
import inspect
import itertools
import operator
import toolz
from toolz.functoolz import (curry, is_valid_args, is_partial_args, is_arity,
from toolz._signatures import builtins
import toolz._signatures as _sigs
from toolz.utils import raises
def test_introspect_builtin_modules():
    mods = [builtins, functools, itertools, operator, toolz, toolz.functoolz, toolz.itertoolz, toolz.dicttoolz, toolz.recipes]
    denylist = set()

    def add_denylist(mod, attr):
        if hasattr(mod, attr):
            denylist.add(getattr(mod, attr))
    add_denylist(builtins, 'basestring')
    add_denylist(builtins, 'NoneType')
    add_denylist(builtins, '__metaclass__')
    add_denylist(builtins, 'sequenceiterator')

    def is_missing(modname, name, func):
        if name.startswith('_') and (not name.startswith('__')):
            return False
        if name.startswith('__pyx_unpickle_') or name.endswith('_cython__'):
            return False
        try:
            if issubclass(func, BaseException):
                return False
        except TypeError:
            pass
        try:
            return callable(func) and func.__module__ is not None and (modname in func.__module__) and (is_partial_args(func, (), {}) is not True) and (func not in denylist)
        except AttributeError:
            return False
    missing = {}
    for mod in mods:
        modname = mod.__name__
        for name, func in vars(mod).items():
            if is_missing(modname, name, func):
                if modname not in missing:
                    missing[modname] = []
                missing[modname].append(name)
    if missing:
        messages = []
        for modname, names in sorted(missing.items()):
            msg = '{}:\n    {}'.format(modname, '\n    '.join(sorted(names)))
            messages.append(msg)
        message = 'Missing introspection for the following callables:\n\n'
        raise AssertionError(message + '\n\n'.join(messages))