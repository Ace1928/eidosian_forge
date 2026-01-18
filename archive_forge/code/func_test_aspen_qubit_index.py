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
def test_aspen_qubit_index():
    """test that AspenQubit properly calculates index and uses it for comparison"""
    qubit10 = AspenQubit(1, 0)
    assert qubit10.index == 10
    assert qubit10 > AspenQubit(0, 5)