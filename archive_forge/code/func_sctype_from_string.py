import builtins
import torch
from . import _dtypes_impl
def sctype_from_string(s):
    """Normalize a string value: a type 'name' or a typecode or a width alias."""
    if s in _names:
        return _names[s]
    if s in _name_aliases.keys():
        return _name_aliases[s]
    if s in _typecodes:
        return _typecodes[s]
    if s in _aliases:
        return _aliases[s]
    if s in _python_types:
        return _python_types[s]
    raise TypeError(f'data type {s!r} not understood')