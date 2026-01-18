from typing import cast, List, Optional, Sequence, Union, Tuple
import numpy as np
from cirq.linalg import tolerance, transformations
from cirq import value
def slice_for_qubits_equal_to(target_qubit_axes: Sequence[int], little_endian_qureg_value: int=0, *, big_endian_qureg_value: int=0, num_qubits: Optional[int]=None, qid_shape: Optional[Tuple[int, ...]]=None) -> Tuple[Union[slice, int, 'ellipsis'], ...]:
    """Returns an index corresponding to a desired subset of an np.ndarray.

    It is assumed that the np.ndarray's shape is of the form (2, 2, 2, ..., 2).

    Example:
        ```python
        # A '4 qubit' tensor with values from 0 to 15.
        r = np.array(range(16)).reshape((2,) * 4)

        # We want to index into the subset where qubit #1 and qubit #3 are ON.
        s = cirq.slice_for_qubits_equal_to([1, 3], 0b11)
        print(s)
        # (slice(None, None, None), 1, slice(None, None, None), 1, Ellipsis)

        # Get that subset. It corresponds to numbers of the form 0b*1*1.
        # where here '*' indicates any possible value.
        print(r[s])
        # [[ 5  7]
        #  [13 15]]
        ```

    Args:
        target_qubit_axes: The qubits that are specified by the index bits. All
            other axes of the slice are unconstrained.
        little_endian_qureg_value: An integer whose bits specify what value is
            desired for of the target qubits. The integer is little endian
            w.r.t. the target qubit axes, meaning the low bit of the integer
            determines the desired value of the first targeted qubit, and so
            forth with the k'th targeted qubit's value set to
            bool(qureg_value & (1 << k)).
        big_endian_qureg_value: Same as `little_endian_qureg_value` but big
            endian w.r.t. to target qubit axes, meaning the low bit of the
            integer dertemines the desired value of the last target qubit, and
            so forth.  Specify exactly one of the `*_qureg_value` arguments.
        num_qubits: If specified the slices will extend all the way up to
            this number of qubits, otherwise if it is None, the final element
            return will be Ellipsis. Optional and defaults to using Ellipsis.
        qid_shape: The qid shape of the state vector being sliced.  Specify this
            instead of `num_qubits` when using qids with dimension != 2.  The
            qureg value is interpreted to store digits with corresponding bases
            packed into an int.

    Returns:
        An index object that will slice out a mutable view of the desired subset
        of a tensor.

    Raises:
        ValueError: If the `qid_shape` mismatches `num_qubits` or exactly one of
            `little_endian_qureg_value` and `big_endian_qureg_value` is not
            specified.
    """
    qid_shape_specified = qid_shape is not None
    if qid_shape is not None or num_qubits is not None:
        if num_qubits is None:
            num_qubits = len(cast(Tuple[int, ...], qid_shape))
        elif qid_shape is None:
            qid_shape = (2,) * num_qubits
        if num_qubits != len(cast(Tuple[int, ...], qid_shape)):
            raise ValueError('len(qid_shape) != num_qubits')
    if little_endian_qureg_value and big_endian_qureg_value:
        raise ValueError('Specify exactly one of the arguments little_endian_qureg_value or big_endian_qureg_value.')
    out_size_specified = num_qubits is not None
    out_size = cast(int, num_qubits) if out_size_specified else max(target_qubit_axes, default=-1) + 1
    result = cast(List[Union[slice, int, 'ellipsis']], [slice(None)] * out_size)
    if not out_size_specified:
        result.append(Ellipsis)
    if qid_shape is None:
        qid_shape = (2,) * out_size
    target_shape = tuple((qid_shape[i] for i in target_qubit_axes))
    if big_endian_qureg_value:
        digits = value.big_endian_int_to_digits(big_endian_qureg_value, base=target_shape)
    else:
        if little_endian_qureg_value < 0 and (not qid_shape_specified):
            little_endian_qureg_value &= (1 << len(target_shape)) - 1
        digits = value.big_endian_int_to_digits(little_endian_qureg_value, base=target_shape[::-1])[::-1]
    for axis, digit in zip(target_qubit_axes, digits):
        result[axis] = digit
    return tuple(result)