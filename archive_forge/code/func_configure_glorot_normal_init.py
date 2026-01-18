from typing import Callable, cast
import numpy
from .backends import Ops
from .config import registry
from .types import FloatsXd, Shape
from .util import partial
@registry.initializers('glorot_normal_init.v1')
def configure_glorot_normal_init() -> Callable[[Shape], FloatsXd]:
    return partial(glorot_normal_init)