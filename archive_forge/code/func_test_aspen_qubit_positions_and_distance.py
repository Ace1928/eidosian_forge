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
def test_aspen_qubit_positions_and_distance():
    """test AspenQubit 2D position and distance calculations"""
    qubit10 = AspenQubit(1, 0)
    assert qubit10.octagon == 1
    assert qubit10.octagon_position == 0
    assert qubit10.dimension == 2
    assert np.isclose(qubit10.x, 3 + 3 / sqrt(2))
    assert np.isclose(qubit10.y, 1 + sqrt(2))
    assert np.isclose(qubit10.distance(AspenQubit(0, 7)), 3 + 2 / sqrt(2))
    assert np.isclose(qubit10.distance(AspenQubit(1, 3)), 1 + sqrt(2))
    qubit15 = AspenQubit(1, 5)
    assert np.isclose(qubit15.x, 2 + sqrt(2))
    qubit15 = AspenQubit(1, 1)
    assert np.isclose(qubit15.x, 2 + sqrt(2) + 1 + sqrt(2))
    with patch('cirq_rigetti.AspenQubit.octagon_position', new_callable=PropertyMock) as mock_octagon_position:
        mock_octagon_position.return_value = 9
        invalid_qubit = AspenQubit(0, 0)
        with pytest.raises(ValueError):
            _ = invalid_qubit.x
        with pytest.raises(ValueError):
            _ = invalid_qubit.y
    with pytest.raises(TypeError):
        _ = qubit10.distance(OctagonalQubit(0))
    with pytest.raises(ValueError):
        _ = AspenQubit(1, 9)