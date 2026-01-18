from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_simple():
    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    class HasApplyMixture:

        def _apply_mixture_(self, args: cirq.ApplyMixtureArgs):
            zero_left = cirq.slice_for_qubits_equal_to(args.left_axes, 0)
            one_left = cirq.slice_for_qubits_equal_to(args.left_axes, 1)
            zero_right = cirq.slice_for_qubits_equal_to(cast(Tuple[int], args.right_axes), 0)
            one_right = cirq.slice_for_qubits_equal_to(cast(Tuple[int], args.right_axes), 1)
            args.out_buffer[:] = 0
            np.copyto(dst=args.auxiliary_buffer0, src=args.target_tensor)
            for kraus_op in [np.sqrt(0.5) * np.eye(2, dtype=np.complex128), np.sqrt(0.5) * x]:
                np.copyto(dst=args.target_tensor, src=args.auxiliary_buffer0)
                cirq.apply_matrix_to_slices(args.target_tensor, kraus_op, [zero_left, one_left], out=args.auxiliary_buffer1)
                cirq.apply_matrix_to_slices(args.auxiliary_buffer1, np.conjugate(kraus_op), [zero_right, one_right], out=args.target_tensor)
                args.out_buffer += args.target_tensor
            return args.out_buffer
    rho = np.copy(x)
    assert_apply_mixture_returns(HasApplyMixture(), rho, [0], [1], assert_result_is_out_buf=True, expected_result=x)