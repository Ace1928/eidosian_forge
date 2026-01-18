from typing import Callable, cast
import numpy
from .backends import Ops
from .config import registry
from .types import FloatsXd, Shape
from .util import partial
@registry.initializers('lecun_uniform_init.v1')
def configure_lecun_uniform_init() -> Callable[[Shape], FloatsXd]:
    return partial(lecun_uniform_init)