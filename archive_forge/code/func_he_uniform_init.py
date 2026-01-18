from typing import Callable, cast
import numpy
from .backends import Ops
from .config import registry
from .types import FloatsXd, Shape
from .util import partial
def he_uniform_init(ops: Ops, shape: Shape) -> FloatsXd:
    scale = numpy.sqrt(6.0 / shape[1])
    return ops.asarray_f(cast(FloatsXd, numpy.random.uniform(-scale, scale, shape)))