import warnings
from typing import (
from torch.onnx import _constants, errors
from torch.onnx._internal import _beartype
@_beartype.beartype
def custom_onnx_symbolic(name: str, opset: Union[OpsetVersion, Sequence[OpsetVersion]], decorate: Optional[Sequence[Callable]]=None) -> Callable:
    """Registers a custom symbolic function.

    Args:
        name: the qualified name of the function.
        opset: the opset version of the function.
        decorate: a sequence of decorators to apply to the function.

    Returns:
        The decorator.

    Raises:
        ValueError: If the separator '::' is not in the name.
    """
    return onnx_symbolic(name, opset, decorate, custom=True)