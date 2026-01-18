import os
from unittest.mock import patch, PropertyMock
from math import sqrt
import pathlib
import json
import pytest
import cirq
from cirq_rigetti import (
from qcs_api_client.models import InstructionSetArchitecture, Node
import numpy as np
def test_aspen_qubit_qid_conversions():
    """test AspenQubit conversion to and from other `cirq.Qid` implementations"""
    qubit10 = AspenQubit(1, 0)
    assert qubit10.to_named_qubit() == cirq.NamedQubit('10')
    assert AspenQubit.from_named_qubit(cirq.NamedQubit('10')) == AspenQubit(1, 0)
    with pytest.raises(ValueError):
        _ = AspenQubit.from_named_qubit(cirq.NamedQubit('s'))
    with pytest.raises(ValueError):
        _ = AspenQubit.from_named_qubit(cirq.NamedQubit('19'))
    with pytest.raises(ValueError):
        _ = qubit10.to_grid_qubit()
    assert AspenQubit(0, 2).to_grid_qubit() == cirq.GridQubit(0, 0)
    assert AspenQubit(0, 1).to_grid_qubit() == cirq.GridQubit(1, 0)
    assert AspenQubit(1, 5).to_grid_qubit() == cirq.GridQubit(0, 1)
    assert AspenQubit(1, 6).to_grid_qubit() == cirq.GridQubit(1, 1)
    assert AspenQubit.from_grid_qubit(cirq.GridQubit(1, 1)) == AspenQubit(1, 6)
    with pytest.raises(ValueError):
        _ = AspenQubit.from_grid_qubit(cirq.GridQubit(3, 4))