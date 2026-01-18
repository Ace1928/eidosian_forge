import array
from numba.core import types
def get_type_class(typ):
    """
    Get the Numba type class for buffer-compatible Python *typ*.
    """
    try:
        return _type_map[typ]
    except KeyError:
        return types.Buffer