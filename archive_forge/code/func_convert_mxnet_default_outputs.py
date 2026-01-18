from typing import Any, Callable, Optional, Tuple, Type
from ..config import registry
from ..model import Model
from ..shims import MXNetShim
from ..types import ArgsKwargs
from ..util import convert_recursive, is_mxnet_array, is_xp_array, mxnet2xp, xp2mxnet
def convert_mxnet_default_outputs(model: Model, X_Ymxnet: Any, is_train: bool):
    X, Ymxnet = X_Ymxnet
    Y = convert_recursive(is_mxnet_array, mxnet2xp, Ymxnet)

    def reverse_conversion(dY: Any) -> ArgsKwargs:
        dYmxnet = convert_recursive(is_xp_array, xp2mxnet, dY)
        return ArgsKwargs(args=((Ymxnet,),), kwargs={'head_grads': dYmxnet})
    return (Y, reverse_conversion)