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
def get_state_token(self):
    """The algorithm is monotonic.  It can only grow or "refine" the
        typevar map.
        """
    return [tv.type for name, tv in sorted(self.typevars.items())]