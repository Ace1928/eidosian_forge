import copy
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from qiskit.assembler.run_config import RunConfig
from qiskit.assembler.assemble_schedules import _assemble_instructions as _assemble_schedule
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classicalregister import Clbit
from qiskit.exceptions import QiskitError
from qiskit.qobj import (
from qiskit.utils.parallel import parallel_map
def index_calibrations() -> Dict[int, List[Tuple[int, GateCalibration]]]:
    """Map each calibration to all experiments that contain it."""
    exp_indices = defaultdict(list)
    for exp_idx, exp in enumerate(experiments):
        for gate_cal in exp.config.calibrations.gates:
            exp_indices[hash(gate_cal)].append((exp_idx, gate_cal))
    return exp_indices