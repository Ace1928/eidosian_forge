from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Union, Tuple
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from pyquil._version import pyquil_version
from pyquil.api import QuantumExecutable
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qvm_client import (
from pyquil.noise import NoiseModel, apply_noise_model
from pyquil.quil import Program, get_classical_addresses_from_program
def check_qvm_version(version: str) -> None:
    """
    Verify that there is no mismatch between pyquil and QVM versions.

    :param version: The version of the QVM
    """
    major, minor, patch = map(int, version.split('.'))
    if major == 1 and minor < 8:
        raise QVMVersionMismatch(f'Must use QVM >= 1.8.0 with pyquil >= 2.8.0, but you have QVM {version} and pyquil {pyquil_version}')