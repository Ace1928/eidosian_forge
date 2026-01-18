import numpy
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, just, tuples
from thinc.api import Linear, NumpyOps
def arrays_BOP_BO(min_B=1, max_B=10, min_O=1, max_O=100, min_P=1, max_P=5):
    shapes = tuples(lengths(lo=min_B, hi=max_B), lengths(lo=min_O, hi=max_O), lengths(lo=min_P, hi=max_P))
    return shapes.flatmap(lambda BOP: tuples(ndarrays_of_shape(BOP), ndarrays_of_shape(BOP[:-1])))