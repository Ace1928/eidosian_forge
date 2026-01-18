from typing import Dict, List, Set, Tuple
import numpy as np
import cirq
import pytest
from cirq.devices.noise_properties import NoiseModelFromNoiseProperties
from cirq.devices.superconducting_qubits_noise_properties import (
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
def default_props(system_qubits: List[cirq.Qid], qubit_pairs: List[Tuple[cirq.Qid, cirq.Qid]]):
    return {'gate_times_ns': DEFAULT_GATE_NS, 't1_ns': {q: 100000.0 for q in system_qubits}, 'tphi_ns': {q: 200000.0 for q in system_qubits}, 'readout_errors': {q: [0.001, 0.01] for q in system_qubits}, 'gate_pauli_errors': {**{OpIdentifier(g, q): 0.001 for g in ExampleNoiseProperties.single_qubit_gates() for q in system_qubits}, **{OpIdentifier(g, q0, q1): 0.01 for g in ExampleNoiseProperties.two_qubit_gates() for q0, q1 in qubit_pairs}}}