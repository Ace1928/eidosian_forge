import torch
from torch import Tensor
from torchmetrics.functional.audio.sdr import scale_invariant_signal_distortion_ratio
from torchmetrics.utilities.checks import _check_same_shape
def scale_invariant_signal_noise_ratio(preds: Tensor, target: Tensor) -> Tensor:
    """`Scale-invariant signal-to-noise ratio`_ (SI-SNR).

    Args:
        preds: float tensor with shape ``(...,time)``
        target: float tensor with shape ``(...,time)``

    Returns:
         Float tensor with shape ``(...,)`` of SI-SNR values per sample

    Raises:
        RuntimeError:
            If ``preds`` and ``target`` does not have the same shape

    Example:
        >>> import torch
        >>> from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio
        >>> target = torch.tensor([3.0, -0.5, 2.0, 7.0])
        >>> preds = torch.tensor([2.5, 0.0, 2.0, 8.0])
        >>> scale_invariant_signal_noise_ratio(preds, target)
        tensor(15.0918)

    """
    return scale_invariant_signal_distortion_ratio(preds=preds, target=target, zero_mean=True)