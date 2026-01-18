from typing import List, cast, Optional, Union, Dict, Any
import functools
from math import sqrt
import httpx
import numpy as np
import networkx as nx
import cirq
from pyquil.quantum_processor import QCSQuantumProcessor
from qcs_api_client.models import InstructionSetArchitecture
from qcs_api_client.operations.sync import get_instruction_set_architecture
from cirq_rigetti._qcs_api_client_decorator import _provide_default_client
@staticmethod
def from_grid_qubit(grid_qubit: cirq.GridQubit) -> 'AspenQubit':
    """Converts `cirq.GridQubit` to `AspenQubit`.

        Returns:
            The equivalent AspenQubit.

        Raises:
            ValueError: GridQubit cannot be converted to AspenQubit.
        """
    if grid_qubit in _grid_qubit_mapping:
        return AspenQubit.from_aspen_index(_grid_qubit_mapping[grid_qubit])
    raise ValueError(f'{grid_qubit} is not convertible to Aspen qubit')