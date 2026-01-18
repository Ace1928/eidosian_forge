from typing import Any, cast, Iterable, Optional, Tuple
import numpy as np
import pytest
import cirq
def test_apply_mixture_no_protocols_implemented_default():

    class NoProtocols:
        pass
    args = cirq.ApplyMixtureArgs(target_tensor=np.eye(2), left_axes=[0], right_axes=[1], out_buffer=None, auxiliary_buffer0=None, auxiliary_buffer1=None)
    result = cirq.apply_mixture(NoProtocols(), args, default='cirq')
    assert result == 'cirq'