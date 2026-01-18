import numpy as np
from numba.pycc import CC
@cc.export('get_const', 'i8()')
def get_const():
    return _const