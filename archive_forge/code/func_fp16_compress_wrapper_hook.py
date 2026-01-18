from typing import Any, Callable
import torch
import torch.distributed as dist
def fp16_compress_wrapper_hook(hook_state, bucket: dist.GradBucket) -> torch.futures.Future[torch.Tensor]:
    bucket.set_buffer(bucket.buffer().to(torch.float16))
    fut = hook(hook_state, bucket)

    def decompress(fut):
        decompressed_tensor = bucket.buffer()
        decompressed_tensor.copy_(fut.value())
        return decompressed_tensor
    return fut.then(decompress)