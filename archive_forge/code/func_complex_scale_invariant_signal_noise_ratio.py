import torch
from torch import Tensor
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
from torchmetrics.utilities.checks import _check_same_shape
def complex_scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor, zero_mean: bool=False) -> Tensor:
    """`Complex scale-invariant signal-to-noise ratio`_ (C-SI-SNR).

    Args:
        preds: real float tensor with shape ``(...,frequency,time,2)`` or complex float tensor with
            shape ``(..., frequency,time)``
        target: real float tensor with shape ``(...,frequency,time,2)`` or complex float tensor with
            shape ``(..., frequency,time)``
        zero_mean: When set to True, the mean of all signals is subtracted prior to computation of the metrics

    Returns:
         Float tensor with shape ``(...,)`` of C-SI-SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` is not the shape (...,frequency,time,2) (after being converted to real if it is complex).
            If ``preds`` and ``target`` does not have the same shape.

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import complex_scale_invariant_signal_noise_ratio
        >>> g = torch.manual_seed(1)
        >>> preds = torch.randn((1,257,100,2))
        >>> target = torch.randn((1,257,100,2))
        >>> complex_scale_invariant_signal_noise_ratio(preds, target)
        tensor([-63.4849])

    """
    if preds.is_complex():
        preds = torch.view_as_real(preds)
    if target.is_complex():
        target = torch.view_as_real(target)
    if (preds.ndim < 3 or preds.shape[-1] != 2) or (target.ndim < 3 or target.shape[-1] != 2):
        raise RuntimeError('Predictions and targets are expected to have the shape (..., frequency, time, 2), but got {preds.shape} and {target.shape}.')
    preds = preds.reshape(*preds.shape[:-3], -1)
    target = target.reshape(*target.shape[:-3], -1)
    return scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=zero_mean)