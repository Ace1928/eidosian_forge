from typing import cast
from ..compat import cupy, tensorflow, torch
from ..types import ArrayXd
from ..util import get_torch_default_device, tensorflow2xp
def cupy_pytorch_allocator(size_in_bytes: int):
    device = get_torch_default_device()
    "Function that can be passed into cupy.cuda.set_allocator, to have cupy\n    allocate memory via PyTorch. This is important when using the two libraries\n    together, as otherwise OOM errors can occur when there's available memory\n    sitting in the other library's pool.\n    "
    size_in_bytes = max(1024, size_in_bytes)
    torch_tensor = torch.zeros((size_in_bytes // 4,), requires_grad=False, device=device)
    address = torch_tensor.data_ptr()
    memory = cupy.cuda.memory.UnownedMemory(address, size_in_bytes, torch_tensor)
    return cupy.cuda.memory.MemoryPointer(memory, 0)