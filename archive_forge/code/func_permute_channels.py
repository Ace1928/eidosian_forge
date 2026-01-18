from typing import List
import PIL.Image
import torch
from torch.nn.functional import conv2d
from torchvision import tv_tensors
from torchvision.transforms import _functional_pil as _FP
from torchvision.transforms._functional_tensor import _max_value
from torchvision.utils import _log_api_usage_once
from ._misc import _num_value_bits, to_dtype_image
from ._type_conversion import pil_to_tensor, to_pil_image
from ._utils import _get_kernel, _register_kernel_internal
def permute_channels(inpt: torch.Tensor, permutation: List[int]) -> torch.Tensor:
    """Permute the channels of the input according to the given permutation.

    This function supports plain :class:`~torch.Tensor`'s, :class:`PIL.Image.Image`'s, and
    :class:`torchvision.tv_tensors.Image` and :class:`torchvision.tv_tensors.Video`.

    Example:
        >>> rgb_image = torch.rand(3, 256, 256)
        >>> bgr_image = F.permutate_channels(rgb_image, permutation=[2, 1, 0])

    Args:
        permutation (List[int]): Valid permutation of the input channel indices. The index of the element determines the
            channel index in the input and the value determines the channel index in the output. For example,
            ``permutation=[2, 0 , 1]``

            - takes ``ìnpt[..., 0, :, :]`` and puts it at ``output[..., 2, :, :]``,
            - takes ``ìnpt[..., 1, :, :]`` and puts it at ``output[..., 0, :, :]``, and
            - takes ``ìnpt[..., 2, :, :]`` and puts it at ``output[..., 1, :, :]``.

    Raises:
        ValueError: If ``len(permutation)`` doesn't match the number of channels in the input.
    """
    if torch.jit.is_scripting():
        return permute_channels_image(inpt, permutation=permutation)
    _log_api_usage_once(permute_channels)
    kernel = _get_kernel(permute_channels, type(inpt))
    return kernel(inpt, permutation=permutation)