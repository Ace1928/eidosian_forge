import numpy as np
import ray
import ray.experimental.array.remote as ra
@ray.remote
def numpy_to_dist(a):
    result = DistArray(a.shape)
    for index in np.ndindex(*result.num_blocks):
        lower = DistArray.compute_block_lower(index, a.shape)
        upper = DistArray.compute_block_upper(index, a.shape)
        idx = tuple((slice(l, u) for l, u in zip(lower, upper)))
        result.object_refs[index] = ray.put(a[idx])
    return result