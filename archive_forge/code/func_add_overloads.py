import traceback
from collections import namedtuple, defaultdict
import itertools
import logging
import textwrap
from shutil import get_terminal_size
from .abstract import Callable, DTypeSpec, Dummy, Literal, Type, weakref
from .common import Opaque
from .misc import unliteral
from numba.core import errors, utils, types, config
from numba.core.typeconv import Conversion
def add_overloads(self, args, qualname, uid):
    """Add an overload of the function.

        Parameters
        ----------
        args :
            argument types
        qualname :
            function qualifying name
        uid :
            unique id
        """
    self._overloads[args] = _RecursiveCallOverloads(qualname, uid)