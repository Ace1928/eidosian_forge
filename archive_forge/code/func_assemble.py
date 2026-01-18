import numpy as np
import ray
import ray.experimental.array.remote as ra
def assemble(self):
    """Assemble an array from a distributed array of object refs."""
    first_block = ray.get(self.object_refs[(0,) * self.ndim])
    dtype = first_block.dtype
    result = np.zeros(self.shape, dtype=dtype)
    for index in np.ndindex(*self.num_blocks):
        lower = DistArray.compute_block_lower(index, self.shape)
        upper = DistArray.compute_block_upper(index, self.shape)
        value = ray.get(self.object_refs[index])
        result[tuple((slice(l, u) for l, u in zip(lower, upper)))] = value
    return result