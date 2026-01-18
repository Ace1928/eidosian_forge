import numpy as np
import pytest
import cirq
def test_apply_channel_no_protocols_implemented_default():

    class NoProtocols:
        pass
    args = cirq.ApplyChannelArgs(target_tensor=np.eye(2), left_axes=[0], right_axes=[1], out_buffer=None, auxiliary_buffer0=None, auxiliary_buffer1=None)
    result = cirq.apply_channel(NoProtocols(), args, 'cirq')
    assert result == 'cirq'