import numpy as np
import ray
import ray.experimental.array.remote as ra
@ray.remote
def subblocks(a, *ranges):
    """
    This function produces a distributed array from a subset of the blocks in
    the `a`. The result and `a` will have the same number of dimensions. For
    example,
        subblocks(a, [0, 1], [2, 4])
    will produce a DistArray whose object_refs are
        [[a.object_refs[0, 2], a.object_refs[0, 4]],
         [a.object_refs[1, 2], a.object_refs[1, 4]]]
    We allow the user to pass in an empty list [] to indicate the full range.
    """
    ranges = list(ranges)
    if len(ranges) != a.ndim:
        raise Exception('sub_blocks expects to receive a number of ranges equal to a.ndim, but it received {} ranges and a.ndim = {}.'.format(len(ranges), a.ndim))
    for i in range(len(ranges)):
        if ranges[i] == []:
            ranges[i] = range(a.num_blocks[i])
        if not np.alltrue(ranges[i] == np.sort(ranges[i])):
            raise Exception('Ranges passed to sub_blocks must be sorted, but the {}th range is {}.'.format(i, ranges[i]))
        if ranges[i][0] < 0:
            raise Exception('Values in the ranges passed to sub_blocks must be at least 0, but the {}th range is {}.'.format(i, ranges[i]))
        if ranges[i][-1] >= a.num_blocks[i]:
            raise Exception('Values in the ranges passed to sub_blocks must be less than the relevant number of blocks, but the {}th range is {}, and a.num_blocks = {}.'.format(i, ranges[i], a.num_blocks))
    last_index = [r[-1] for r in ranges]
    last_block_shape = DistArray.compute_block_shape(last_index, a.shape)
    shape = [(len(ranges[i]) - 1) * BLOCK_SIZE + last_block_shape[i] for i in range(a.ndim)]
    result = DistArray(shape)
    for index in np.ndindex(*result.num_blocks):
        result.object_refs[index] = a.object_refs[tuple((ranges[i][index[i]] for i in range(a.ndim)))]
    return result