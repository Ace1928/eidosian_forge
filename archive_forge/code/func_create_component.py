from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar
import srsly
from ..compat import tensorflow as tf
from ..model import Model
from ..shims import TensorFlowShim, keras_model_fns, maybe_handshake_model
from ..types import ArgsKwargs, ArrayXd
from ..util import (
@keras_model_fns(name)
def create_component(*call_args, **call_kwargs):
    return clazz(*call_args, **call_kwargs)