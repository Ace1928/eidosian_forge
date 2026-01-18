from dataclasses import dataclass
import uuid
from collections import defaultdict
from typing import Dict, List, Optional, Union, cast
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import ParameterAref, ParameterSpec
from pyquil.api import QuantumExecutable, EncryptedProgram, EngagementManager
from pyquil._memory import Memory
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qpu_client import GetBuffersRequest, QPUClient, BufferResponse, RunProgramRequest
from pyquil.quilatom import (
@property
def quantum_processor_id(self) -> str:
    """ID of quantum processor targeted."""
    return self._qpu_client.quantum_processor_id