import numpy
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers, just, tuples
from thinc.api import Linear, NumpyOps
def arrays_OPFI_BI_lengths(max_B=5, max_P=3, max_F=5, max_I=8):
    shapes = tuples(lengths(hi=max_B), lengths(hi=max_P), lengths(hi=max_F), lengths(hi=max_I), arrays('int32', shape=(5,), elements=integers(min_value=1, max_value=10)))
    strat = shapes.flatmap(lambda opfi_lengths: tuples(ndarrays_of_shape(opfi_lengths[:-1]), ndarrays_of_shape((sum(opfi_lengths[-1]), opfi_lengths[-2])), just(opfi_lengths[-1])))
    return strat