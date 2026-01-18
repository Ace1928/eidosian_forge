from typing import Any, Callable, Dict, Optional, Tuple, Type, TypeVar
import srsly
from ..compat import tensorflow as tf
from ..model import Model
from ..shims import TensorFlowShim, keras_model_fns, maybe_handshake_model
from ..types import ArgsKwargs, ArrayXd
from ..util import (
def keras_subclass(name: str, X: XType, Y: YType, input_shape: Tuple[int, ...], compile_args: Optional[Dict[str, Any]]=None) -> Callable[[InFunc], InFunc]:
    """Decorate a custom keras subclassed model with enough information to
    serialize and deserialize it reliably in the face of the many restrictions
    on keras subclassed models.

    name (str): The unique namespace string to use to represent this model class.
    X (Any): A sample X input for performing a forward pass on the network.
    Y (Any): A sample Y input for performing a backward pass on the network.
    input_shape (Tuple[int, ...]): A set of input shapes for building the network.
    compile: Arguments to pass directly to the keras `model.compile` call.

    RETURNS (Callable): The decorated class.
    """
    compile_defaults = {'optimizer': 'adam', 'loss': 'mse'}
    if compile_args is None:
        compile_args = compile_defaults
    else:
        compile_args = {**compile_defaults, **compile_args}

    def call_fn(clazz):
        clazz.catalogue_name = property(lambda inst: name)
        clazz.eg_shape = property(lambda inst: input_shape)
        clazz.eg_compile = property(lambda inst: compile_args)
        clazz.eg_x = property(lambda inst: X)
        clazz.eg_y = property(lambda inst: Y)

        @keras_model_fns(name)
        def create_component(*call_args, **call_kwargs):
            return clazz(*call_args, **call_kwargs)
        wrapped_init = clazz.__init__

        def __init__(self, *args, **kwargs):
            wrapped_init(self, *args, **kwargs)
            try:
                srsly.json_dumps(args)
                srsly.json_dumps(kwargs)
            except BaseException as _err:
                raise ValueError(f'In order to serialize Keras Subclass models, the constructor arguments must be serializable. This allows thinc to recreate the code-based model with the same configuration.\nThe encountered error is: {_err}')
            self.eg_args = ArgsKwargs(args, kwargs)
        clazz.__init__ = __init__
        return clazz
    return call_fn