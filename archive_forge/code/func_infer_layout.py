import array
from numba.core import types
def infer_layout(val):
    """
    Infer layout of the given memoryview *val*.
    """
    return 'C' if val.c_contiguous else 'F' if val.f_contiguous else 'A'