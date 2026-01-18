import itertools
import re
import socket
import subprocess
import warnings
from contextlib import contextmanager
from math import pi, log
from typing import (
import httpx
import networkx as nx
import numpy as np
from qcs_api_client.client import QCSClientConfiguration
from qcs_api_client.models import ListQuantumProcessorsResponse
from qcs_api_client.operations.sync import list_quantum_processors
from rpcq.messages import ParameterAref
from pyquil.api import EngagementManager
from pyquil.api._abstract_compiler import AbstractCompiler, QuantumExecutable
from pyquil.api._compiler import QPUCompiler, QVMCompiler
from pyquil.api._qam import QAM, QAMExecutionResult
from pyquil.api._qcs_client import qcs_client
from pyquil.api._qpu import QPU
from pyquil.api._qvm import QVM
from pyquil.experiment._main import Experiment
from pyquil.experiment._memory import merge_memory_map_lists
from pyquil.experiment._result import ExperimentResult, bitstrings_to_expectations
from pyquil.experiment._setting import ExperimentSetting
from pyquil.external.rpcq import CompilerISA
from pyquil.gates import RX, MEASURE
from pyquil.noise import decoherence_noise_with_asymmetric_ro, NoiseModel
from pyquil.paulis import PauliTerm
from pyquil.pyqvm import PyQVM
from pyquil.quantum_processor import (
from pyquil.quil import Program
def list_quantum_computers(qpus: bool=True, qvms: bool=True, timeout: float=10.0, client_configuration: Optional[QCSClientConfiguration]=None) -> List[str]:
    """
    List the names of available quantum computers

    :param qpus: Whether to include QPUs in the list.
    :param qvms: Whether to include QVMs in the list.
    :param timeout: Time limit for request, in seconds.
    :param client_configuration: Optional client configuration. If none is provided, a default one will be loaded.
    """
    client_configuration = client_configuration or QCSClientConfiguration.load()
    qc_names: List[str] = []
    if qpus:
        with qcs_client(client_configuration=client_configuration, request_timeout=timeout) as client:
            qcs: ListQuantumProcessorsResponse = list_quantum_processors(client=client, page_size=100).parsed
        qc_names += [qc.id for qc in qcs.quantum_processors]
    if qvms:
        qc_names += ['9q-square-qvm', '9q-square-noisy-qvm']
    return qc_names