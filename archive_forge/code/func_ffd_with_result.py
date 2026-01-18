import numpy as np
import numba
@numba.njit
def ffd_with_result(a: np.ndarray, c: int, start_index: int):
    indices = np.argsort(a)[::-1]
    a = a[indices]
    bins = []
    bins_result = []
    for a_id, size in enumerate(a):
        add_new = True
        for idx in range(len(bins)):
            if bins[idx] >= size:
                bins[idx] -= size
                bins_result[idx].append(indices[a_id] + start_index)
                add_new = False
                break
        if add_new:
            bins.append(c - size)
            bins_result.append([indices[a_id] + start_index])
    return bins_result