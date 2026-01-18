from typing import List, Union
from ray.rllib.utils.framework import try_import_torch
def make_time_major(tensor: Union['torch.Tensor', List['torch.Tensor']], *, trajectory_len: int=None, recurrent_seq_len: int=None):
    """Swaps batch and trajectory axis.

    Args:
        tensor: A tensor or list of tensors to swap the axis of.
            NOTE: Each tensor must have the shape [B * T] where B is the batch size and
            T is the trajectory length.
        trajectory_len: The length of each trajectory being transformed.
            If None then `recurrent_seq_len` must be set.
        recurrent_seq_len: Sequence lengths if recurrent.
            If None then `trajectory_len` must be set.

    Returns:
        res: A tensor with swapped axes or a list of tensors with
        swapped axes.
    """
    if isinstance(tensor, (list, tuple)):
        return [make_time_major(_tensor, trajectory_len, recurrent_seq_len) for _tensor in tensor]
    assert trajectory_len != recurrent_seq_len and (trajectory_len is None or recurrent_seq_len is None), 'Either trajectory_len or recurrent_seq_len must be set.'
    if recurrent_seq_len:
        B = recurrent_seq_len.shape[0]
        T = tensor.shape[0] // B
    else:
        T = trajectory_len
        B = tensor.shape[0] // T
    rs = torch.reshape(tensor, [B, T] + list(tensor.shape[1:]))
    res = torch.transpose(rs, 1, 0)
    return res