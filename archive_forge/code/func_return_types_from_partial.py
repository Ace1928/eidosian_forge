import logging
import operator
import contextlib
import itertools
from pprint import pprint
from collections import OrderedDict, defaultdict
from functools import reduce
from numba.core import types, utils, typing, ir, config
from numba.core.typing.templates import Signature
from numba.core.errors import (TypingError, UntypedAttributeError,
from numba.core.funcdesc import qualifying_prefix
from numba.core.typeconv import Conversion
def return_types_from_partial(self):
    """
        Resume type inference partially to deduce the return type.
        Note: No side-effect to `self`.

        Returns the inferred return type or None if it cannot deduce the return
        type.
        """
    cloned = self.copy(skip_recursion=True)
    cloned.build_constraint()
    cloned.propagate(raise_errors=False)
    rettypes = set()
    for retvar in cloned._get_return_vars():
        if retvar.name in cloned.typevars:
            typevar = cloned.typevars[retvar.name]
            if typevar and typevar.defined:
                rettypes.add(types.unliteral(typevar.getone()))
    if not rettypes:
        return
    return cloned._unify_return_types(rettypes)