import numpy as np
import pytest
import cirq
def test_apply_channel_returns_aux_buffer():
    rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)

    class ReturnsAuxBuffer0:

        def _apply_channel_(self, args: cirq.ApplyChannelArgs):
            return args.auxiliary_buffer0
    with pytest.raises(AssertionError, match='ReturnsAuxBuffer0'):
        _ = apply_channel(ReturnsAuxBuffer0(), rho, [0], [1])

    class ReturnsAuxBuffer1:

        def _apply_channel_(self, args: cirq.ApplyChannelArgs):
            return args.auxiliary_buffer1
    with pytest.raises(AssertionError, match='ReturnsAuxBuffer1'):
        _ = apply_channel(ReturnsAuxBuffer1(), rho, [0], [1])