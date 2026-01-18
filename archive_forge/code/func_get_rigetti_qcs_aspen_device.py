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
@_provide_default_client
def get_rigetti_qcs_aspen_device(quantum_processor_id: str, client: Optional[httpx.Client]) -> RigettiQCSAspenDevice:
    """Retrieves a `qcs_api_client.models.InstructionSetArchitecture` from the Rigetti
    QCS API and uses it to initialize a RigettiQCSAspenDevice.

    Args:
        quantum_processor_id: The identifier of the Rigetti QCS quantum processor.
        client: Optional; A `httpx.Client` initialized with Rigetti QCS credentials
        and configuration. If not provided, `qcs_api_client` will initialize a
        configured client based on configured values in the current user's
        `~/.qcs` directory or default values.

    Returns:
        A `RigettiQCSAspenDevice` with the specified quantum processor instruction
        set and architecture.

    """
    isa = cast(InstructionSetArchitecture, get_instruction_set_architecture(client=client, quantum_processor_id=quantum_processor_id).parsed)
    return RigettiQCSAspenDevice(isa=isa)