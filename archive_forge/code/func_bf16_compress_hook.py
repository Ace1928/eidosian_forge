from typing import Any, Callable
import torch
import torch.distributed as dist
def bf16_compress_hook(process_group: dist.ProcessGroup, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This DDP communication hook implements a simple gradient compression
    approach that casts ``GradBucket`` tensor to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format>`_ (``torch.bfloat16``)
    and then divides it by the process group size.
    It allreduces those ``bfloat16`` gradient tensors. Once compressed gradient
    tensors are allreduced, the chained callback ``decompress`` casts it back to the input data type (such as ``float32``).

    Example::
        >>> # xdoctest: +SKIP
        >>> ddp_model.register_comm_hook(process_group, bf16_compress_hook)
    """
    group_to_use = process_group if process_group is not None else dist.group.WORLD
    world_size = group_to_use.size()
    compressed_tensor = bucket.buffer().to(torch.bfloat16).div_(world_size)
    fut = dist.all_reduce(compressed_tensor, group=group_to_use, async_op=True).get_future()

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        decompressed_tensor.copy_(fut.value()[0])
        return decompressed_tensor
    return fut.then(decompress)