from typing import cast
from ..compat import cupy, tensorflow, torch
from ..types import ArrayXd
from ..util import get_torch_default_device, tensorflow2xp
def cupy_tensorflow_allocator(size_in_bytes: int):
    """Function that can be passed into cupy.cuda.set_allocator, to have cupy
    allocate memory via TensorFlow. This is important when using the two libraries
    together, as otherwise OOM errors can occur when there's available memory
    sitting in the other library's pool.
    """
    size_in_bytes = max(1024, size_in_bytes)
    tensor = tensorflow.zeros((size_in_bytes // 4,), dtype=tensorflow.dtypes.float32)
    cupy_array = cast(ArrayXd, tensorflow2xp(tensor))
    address = int(cupy_array.data)
    memory = cupy.cuda.memory.UnownedMemory(address, size_in_bytes, cupy_array)
    return cupy.cuda.memory.MemoryPointer(memory, 0)