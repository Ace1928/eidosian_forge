import numpy
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, just, tuples
from thinc.api import Linear, NumpyOps
def arrays_BI_BO(min_B=1, max_B=10, min_I=1, max_I=100, min_O=1, max_O=100):
    shapes = tuples(lengths(lo=min_B, hi=max_B), lengths(lo=min_I, hi=max_I), lengths(lo=min_O, hi=max_O))
    return shapes.flatmap(lambda BIO: tuples(ndarrays_of_shape((BIO[0], BIO[1])), ndarrays_of_shape((BIO[0], BIO[2]))))