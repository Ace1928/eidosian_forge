import dataclasses
import typing
from typing import Callable, Optional
import cupy
from cupyx.distributed.array import _array
from cupyx.distributed.array import _chunk
from cupyx.distributed.array import _modes
def make_2d_index_map(i_partitions: list[int], j_partitions: list[int], devices: list[list[set[int]]]) -> dict[int, list[tuple[slice, ...]]]:
    """Create an ``index_map`` for a 2D matrix with a specified blocking.

    Args:
        i_partitions (list of ints): boundaries of blocks on the `i` axis
        j_partitions (list of ints): boundaries of blocks on the `j` axis
        devices (2D list of sets of ints): devices owning each block

    Returns:
        dict from int to array indices: index_map
            Indices for the chunks that devices with designated IDs are going
            to own.

    Example:
        >>> index_map = make_2d_index_map(
        ...     [0, 2, 4], [0, 3, 5],
        ...     [[{0}, {1}],
        ...      [{2}, {0, 1}]])
        >>> pprint(index_map)
        {0: [(slice(0, 2, None), slice(0, 3, None)),
             (slice(2, 4, None), slice(3, 5, None))],
         1: [(slice(0, 2, None), slice(3, 5, None)),
             (slice(2, 4, None), slice(3, 5, None))],
         2: [(slice(2, 4, None), slice(0, 3, None))]}
    """
    assert i_partitions[0] == 0
    assert sorted(set(i_partitions)) == i_partitions
    assert j_partitions[0] == 0
    assert sorted(set(j_partitions)) == j_partitions
    index_map: dict[int, list[tuple[slice, ...]]] = {}
    assert len(devices) == len(i_partitions) - 1
    for i in range(len(devices)):
        assert len(devices[i]) == len(j_partitions) - 1
        for j in range(len(devices[i])):
            i_start = i_partitions[i]
            i_stop = i_partitions[i + 1]
            j_start = j_partitions[j]
            j_stop = j_partitions[j + 1]
            idx = (slice(i_start, i_stop), slice(j_start, j_stop))
            for dev in devices[i][j]:
                index_map.setdefault(dev, []).append(idx)
    return index_map