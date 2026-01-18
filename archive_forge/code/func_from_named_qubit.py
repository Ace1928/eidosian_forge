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
def from_named_qubit(qubit: cirq.NamedQubit) -> 'AspenQubit':
    """Converts `cirq.NamedQubit` to `AspenQubit`.

        Returns:
            The equivalent AspenQubit.

        Raises:
            ValueError: NamedQubit cannot be converted to AspenQubit.
            UnsupportedQubit: If the supplied qubit is not a named qubit with an octagonal
                index.
        """
    try:
        index = int(qubit.name)
        return AspenQubit.from_aspen_index(index)
    except ValueError:
        raise UnsupportedQubit('Aspen devices only support named qubits by octagonal index')