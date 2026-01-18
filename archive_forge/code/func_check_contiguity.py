import numpy as np
import operator
from collections import namedtuple
from numba.core import types, utils
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.core.typing import collections
from numba.core.errors import (TypingError, RequireLiteralValue, NumbaTypeError,
from numba.core.cgutils import is_nonelike
def check_contiguity(outer_indices):
    """
            Whether indexing with the given indices (from outer to inner in
            physical layout order) can keep an array contiguous.
            """
    for ty in outer_indices[:-1]:
        if not keeps_contiguity(ty, False):
            return False
    if outer_indices and (not keeps_contiguity(outer_indices[-1], True)):
        return False
    return True