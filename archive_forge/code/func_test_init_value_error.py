import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('pauli,flip_x,flip_z', itertools.product(_paulis, _bools, _bools))
def test_init_value_error(pauli, flip_x, flip_z):
    with pytest.raises(ValueError):
        cirq.SingleQubitCliffordGate.from_xz_map((pauli, flip_x), (pauli, flip_z))