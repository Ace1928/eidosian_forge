from typing import List, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from typing_extensions import Literal
from torchmetrics.functional.image.utils import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d
from torchmetrics.utilities.checks import _check_same_shape
from torchmetrics.utilities.distributed import reduce
def structural_similarity_index_measure(preds: Tensor, target: Tensor, gaussian_kernel: bool=True, sigma: Union[float, Sequence[float]]=1.5, kernel_size: Union[int, Sequence[int]]=11, reduction: Literal['elementwise_mean', 'sum', 'none', None]='elementwise_mean', data_range: Optional[Union[float, Tuple[float, float]]]=None, k1: float=0.01, k2: float=0.03, return_full_image: bool=False, return_contrast_sensitivity: bool=False) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute Structural Similarity Index Measure.

    Args:
        preds: estimated image
        target: ground truth image
        gaussian_kernel: If true (default), a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exclusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the constant term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``

    Return:
        Tensor with SSIM score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If the length of ``kernel_size`` or ``sigma`` is not ``2``.
        ValueError:
            If one of the elements of ``kernel_size`` is not an ``odd positive number``.
        ValueError:
            If one of the elements of ``sigma`` is not a ``positive number``.

    Example:
        >>> from torchmetrics.functional.image import structural_similarity_index_measure
        >>> preds = torch.rand([3, 3, 256, 256])
        >>> target = preds * 0.75
        >>> structural_similarity_index_measure(preds, target)
        tensor(0.9219)

    """
    preds, target = _ssim_check_inputs(preds, target)
    similarity_pack = _ssim_update(preds, target, gaussian_kernel, sigma, kernel_size, data_range, k1, k2, return_full_image, return_contrast_sensitivity)
    if isinstance(similarity_pack, tuple):
        similarity, image = similarity_pack
        return (_ssim_compute(similarity, reduction), image)
    similarity = similarity_pack
    return _ssim_compute(similarity, reduction)