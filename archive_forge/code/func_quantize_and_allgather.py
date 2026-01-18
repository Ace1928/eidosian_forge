import torch
import torch.distributed as dist
from torch import nn
def quantize_and_allgather(fut):
    all_ranks_s_and_z = fut.wait()[0]
    quantized_tensor = _quantize_per_channel_cuda(tensor_in_channels, all_ranks_s_and_z[rank, 0, :], all_ranks_s_and_z[rank, 1, :])
    fut = dist.all_gather(_get_allgather_out_list(quantized_tensor, world_size), quantized_tensor, group=group_to_use, async_op=True).get_future()
    return fut.wait()