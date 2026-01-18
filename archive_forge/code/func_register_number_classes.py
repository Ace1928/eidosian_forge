import warnings
import numpy as np
import operator
from numba.core import types, utils, config
from numba.core.typing.templates import (AttributeTemplate, AbstractTemplate,
from numba.np.numpy_support import (ufunc_find_matching_loop,
from numba.core.errors import (TypingError, NumbaPerformanceWarning,
from numba import pndindex
def register_number_classes(register_global):
    for np_type in np_types:
        nb_type = getattr(types, np_type.__name__)
        register_global(np_type, types.NumberClass(nb_type))