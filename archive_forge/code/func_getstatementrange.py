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
def getstatementrange(self, lineno: int) -> Tuple[int, int]:
    """Return (start, end) tuple which spans the minimal statement region
        which containing the given lineno."""
    if not 0 <= lineno < len(self):
        raise IndexError('lineno out of range')
    ast, start, end = getstatementrange_ast(lineno, self)
    return (start, end)