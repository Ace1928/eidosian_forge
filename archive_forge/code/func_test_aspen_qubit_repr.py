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
def test_aspen_qubit_repr():
    """test AspenQubit.__repr__"""
    qubit10 = AspenQubit(1, 0)
    assert 'cirq_rigetti.AspenQubit(octagon=1, octagon_position=0)' == repr(qubit10)