from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar
import srsly
from ..compat import tensorflow as tf
from ..model import Model
from ..shims import TensorFlowShim, keras_model_fns, maybe_handshake_model
from ..types import ArgsKwargs, ArrayXd
from ..util import (
def reverse_conversion(dXtf):
    dX = convert_recursive(is_tensorflow_array, tensorflow2xp, dXtf)
    return dX.args[0]