import ast
from bisect import bisect_right
import inspect
import textwrap
import tokenize
import types
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Tuple
from typing import Union
import warnings
def getrawcode(obj: object, trycall: bool=True) -> types.CodeType:
    """Return code object for given function."""
    try:
        return obj.__code__
    except AttributeError:
        pass
    if trycall:
        call = getattr(obj, '__call__', None)
        if call and (not isinstance(obj, type)):
            return getrawcode(call, trycall=False)
    raise TypeError(f'could not get code object for {obj!r}')