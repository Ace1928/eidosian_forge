from typing import Callable, cast
import numpy
from .backends import Ops
from .config import registry
from .types import FloatsXd, Shape
from .util import partial
def normal_init(ops: Ops, shape: Shape, *, mean: float=0) -> FloatsXd:
    size = int(ops.xp.prod(ops.xp.asarray(shape)))
    inits = cast(FloatsXd, numpy.random.normal(scale=mean, size=size).astype('float32'))
    inits = ops.reshape_f(inits, shape)
    return ops.asarray_f(inits)