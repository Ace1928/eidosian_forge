from .cu2qu import *
import random
import timeit
def setup_curves_to_quadratic():
    num_curves = 3
    return ([generate_curve() for curve in range(num_curves)], [MAX_ERR] * num_curves)