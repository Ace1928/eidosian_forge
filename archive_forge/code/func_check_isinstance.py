import functools
import itertools
import re
import sys
import warnings
from .deprecation import (
def check_isinstance(types, /, **kwargs):
    """
    For each *key, value* pair in *kwargs*, check that *value* is an instance
    of one of *types*; if not, raise an appropriate TypeError.

    As a special case, a ``None`` entry in *types* is treated as NoneType.

    Examples
    --------
    >>> _api.check_isinstance((SomeClass, None), arg=arg)
    """
    none_type = type(None)
    types = (types,) if isinstance(types, type) else (none_type,) if types is None else tuple((none_type if tp is None else tp for tp in types))

    def type_name(tp):
        return 'None' if tp is none_type else tp.__qualname__ if tp.__module__ == 'builtins' else f'{tp.__module__}.{tp.__qualname__}'
    for k, v in kwargs.items():
        if not isinstance(v, types):
            names = [*map(type_name, types)]
            if 'None' in names:
                names.remove('None')
                names.append('None')
            raise TypeError('{!r} must be an instance of {}, not a {}'.format(k, ', '.join(names[:-1]) + ' or ' + names[-1] if len(names) > 1 else names[0], type_name(type(v))))