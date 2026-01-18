import numpy as np
import pytest
import cirq
def test_apply_channel_apply_unitary():
    shape = (2, 2, 2, 2)
    rho = np.ones(shape, dtype=np.complex128)

    class HasApplyUnitaryOutputInBuffer:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            zero = args.subspace_index(0)
            one = args.subspace_index(1)
            args.available_buffer[zero] = args.target_tensor[zero]
            args.available_buffer[one] = 1j * args.target_tensor[one]
            return args.available_buffer

    class HasApplyUnitaryMutateInline:

        def _apply_unitary_(self, args: cirq.ApplyUnitaryArgs) -> np.ndarray:
            one = args.subspace_index(1)
            args.target_tensor[one] *= 1j
            return args.target_tensor
    for val in (HasApplyUnitaryOutputInBuffer(), HasApplyUnitaryMutateInline()):
        result_is_out_buf = isinstance(val, HasApplyUnitaryOutputInBuffer)
        result = apply_channel(val, rho, left_axes=[1], right_axes=[3], assert_result_is_out_buf=result_is_out_buf)
        np.testing.assert_almost_equal(result, np.reshape(np.outer([1, 1j, 1, 1j], [1, -1j, 1, -1j]), shape))