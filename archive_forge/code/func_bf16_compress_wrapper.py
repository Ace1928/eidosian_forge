from typing import Any, Callable
import torch
import torch.distributed as dist
def bf16_compress_wrapper(hook: Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]) -> Callable[[Any, dist.GradBucket], torch.futures.Future[torch.Tensor]]:
    """
    Warning: This API is experimental, and it requires NCCL version later than 2.9.6.

    This wrapper casts the input gradient tensor of a given DDP communication hook to half-precision
    `Brain floating point format <https://en.wikipedia.org/wiki/Bfloat16_floating-point_format> `_  (``torch.bfloat16``),
    and casts the resulting tensor of the given hook back to the input data type, such as ``float32``.

    Therefore, ``bf16_compress_hook`` is equivalent to ``bf16_compress_wrapper(allreduce_hook)``.

    Example::
        >>> # xdoctest: +SKIP
        >>> state = PowerSGDState(process_group=process_group, matrix_approximation_rank=1, start_powerSGD_iter=10)
        >>> ddp_model.register_comm_hook(state, bf16_compress_wrapper(powerSGD_hook))
    """

    def bf16_compress_wrapper_hook(hook_state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
        bucket.set_buffer(bucket.buffer().to(torch.bfloat16))
        fut = hook(hook_state, bucket)

        def decompress(fut):
            decompressed_tensor = bucket.buffer()
            decompressed_tensor.copy_(fut.value())
            return decompressed_tensor
        return fut.then(decompress)
    return bf16_compress_wrapper_hook