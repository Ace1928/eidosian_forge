from contextlib import contextmanager
import inspect
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.logic.boolalg import Boolean, false, true
from sympy.multipledispatch.dispatcher import Dispatcher, str_signature
from sympy.utilities.exceptions import sympy_deprecation_warning
from sympy.utilities.iterables import is_sequence
from sympy.utilities.source import get_class
@classmethod
def register_many(cls, *types, **kwargs):
    """
        Register multiple signatures to same handler.
        """

    def _(func):
        for t in types:
            if not is_sequence(t):
                t = (t,)
            cls.register(*t, **kwargs)(func)
    return _