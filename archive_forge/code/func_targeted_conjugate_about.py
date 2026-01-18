import dataclasses
from typing import Any, List, Optional, Sequence, Tuple, Union
import numpy as np
from cirq import protocols
from cirq.linalg import predicates
def targeted_conjugate_about(tensor: np.ndarray, target: np.ndarray, indices: Sequence[int], conj_indices: Optional[Sequence[int]]=None, buffer: Optional[np.ndarray]=None, out: Optional[np.ndarray]=None) -> np.ndarray:
    """Conjugates the given tensor about the target tensor.

    This method computes a target tensor conjugated by another tensor.
    Here conjugate is used in the sense of conjugating by a matrix, i.a.
    A conjugated about B is $A B A^\\dagger$ where $\\dagger$ represents the
    conjugate transpose.

    Abstractly this compute $A \\cdot B \\cdot A^\\dagger$ where A and B are
    multi-dimensional arrays, and instead of matrix multiplication $\\cdot$
    is a contraction between the given indices (indices for first $\\cdot$,
    conj_indices for second $\\cdot$).

    More specifically, this computes:

    $$
    \\sum tensor_{i_0,...,i_{r-1},j_0,...,j_{r-1}} *
        target_{k_0,...,k_{r-1},l_0,...,l_{r-1}} *
        tensor_{m_0,...,m_{r-1},n_0,...,n_{r-1}}^*
    $$

    where the sum is over indices where $j_s$ = $k_s$ and $s$ is in `indices`
    and $l_s$ = $m_s$ and s is in `conj_indices`.

    Args:
        tensor: The tensor that will be conjugated about the target tensor.
        target: The tensor that will receive the conjugation.
        indices: The indices which will be contracted between the tensor and
            target.
        conj_indices: The indices which will be contracted between the
            complex conjugate of the tensor and the target. If this is None,
            then these will be the values in indices plus half the number
            of dimensions of the target (`ndim`). This is the most common case
            and corresponds to the case where the target is an operator on
            a n-dimensional tensor product space (here `n` would be `ndim`).
        buffer: A buffer to store partial results in.  If not specified or None,
            a new buffer is used.
        out: The buffer to store the results in. If not specified or None, a new
            buffer is used. Must have the same shape as target.

    Returns:
        The result of the conjugation, as a numpy array.
    """
    conj_indices = conj_indices or [i + target.ndim // 2 for i in indices]
    first_multiply = targeted_left_multiply(tensor, target, indices, out=buffer)
    return targeted_left_multiply(np.conjugate(tensor), first_multiply, conj_indices, out=out)