import datetime
import pytest
import matplotlib as mpl
import matplotlib.pyplot as plt
from google.protobuf.text_format import Merge
import cirq
import cirq_google as cg
from cirq_google.api import v2
def test_key_to_qubits():
    qubits = tuple([cirq.GridQubit(1, 1), cirq.GridQubit(1, 2)])
    assert cg.Calibration.key_to_qubit(qubits) == cirq.GridQubit(1, 1)
    assert cg.Calibration.key_to_qubits(qubits) == (cirq.GridQubit(1, 1), cirq.GridQubit(1, 2))
    with pytest.raises(ValueError, match='was not a tuple of qubits'):
        cg.Calibration.key_to_qubit('alpha')
    with pytest.raises(ValueError, match='was not a tuple of grid qubits'):
        cg.Calibration.key_to_qubits('alpha')