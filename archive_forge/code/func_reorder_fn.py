from typing import Sequence, Callable
from functools import reduce
import pennylane as qml
from pennylane.transforms import transform
def reorder_fn(res):
    """re-order the output to the original shape and order"""
    if len(tapes[0].measurements) == 1:
        shot_vector_defined = isinstance(res[0], tuple)
    else:
        shot_vector_defined = isinstance(res[0][0], tuple)
    res = list(zip(*res)) if shot_vector_defined else [res]
    reorder_indxs = qml.math.concatenate(group_coeffs)
    res_ordered = []
    for shot_res in res:
        shot_res = reduce(lambda x, y: x + list(y) if isinstance(y, (tuple, list)) else x + [y], shot_res, [])
        shot_res = list(zip(range(len(shot_res)), shot_res))
        shot_res = sorted(shot_res, key=lambda r: reorder_indxs[r[0]])
        shot_res = [r[1] for r in shot_res]
        res_ordered.append(tuple(shot_res))
    return tuple(res_ordered) if shot_vector_defined else res_ordered[0]