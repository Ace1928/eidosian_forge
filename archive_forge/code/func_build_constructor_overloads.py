from typing import List, Tuple, Dict
from numba import types
from numba.core import cgutils
from numba.core.extending import make_attribute_wrapper, models, register_model
from numba.core.imputils import Registry as ImplRegistry
from numba.core.typing.templates import ConcreteTemplate
from numba.core.typing.templates import Registry as TypingRegistry
from numba.core.typing.templates import signature
from numba.cuda import stubs
from numba.cuda.errors import CudaLoweringError
def build_constructor_overloads(base_type, vty_name, num_elements, arglists, l):
    """
    For a given vector type, build a list of overloads for its constructor.
    """
    if num_elements == 0:
        arglists.append(l[:])
    for i in range(1, num_elements + 1):
        if i == 1:
            l.append(base_type)
            build_constructor_overloads(base_type, vty_name, num_elements - i, arglists, l)
            l.pop(-1)
            l.append(vector_types[f'{vty_name[:-1]}1'])
            build_constructor_overloads(base_type, vty_name, num_elements - i, arglists, l)
            l.pop(-1)
        else:
            l.append(vector_types[f'{vty_name[:-1]}{i}'])
            build_constructor_overloads(base_type, vty_name, num_elements - i, arglists, l)
            l.pop(-1)