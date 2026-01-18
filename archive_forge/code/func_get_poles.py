import functools
import math
import operator
import textwrap
import cupy
def get_poles(order):
    if order == 2:
        return (-0.1715728752538099,)
    elif order == 3:
        return (-0.2679491924311227,)
    elif order == 4:
        return (-0.36134122590022016, -0.013725429297339121)
    elif order == 5:
        return (-0.4305753470999738, -0.04309628820326465)
    else:
        raise ValueError('only order 2-5 supported')