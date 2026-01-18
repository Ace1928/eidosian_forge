import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def parameterize_circuit(circuit: 'cirq.Circuit', options: XEBCharacterizationOptions) -> 'cirq.Circuit':
    """Parameterize PhasedFSim-like gates in a given circuit according to
    `phased_fsim_options`.
    """
    gate = options.get_parameterized_gate()
    return circuits.Circuit((circuits.Moment((gate.on(*op.qubits) if options.should_parameterize(op) else op for op in moment.operations)) for moment in circuit.moments))