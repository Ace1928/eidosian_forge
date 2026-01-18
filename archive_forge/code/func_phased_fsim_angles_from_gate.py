import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def phased_fsim_angles_from_gate(gate: 'cirq.Gate') -> Dict[str, 'cirq.TParamVal']:
    """For a given gate, return a dictionary mapping '{angle}_default' to its noiseless value
    for the five PhasedFSim angles."""
    defaults: Dict[str, 'cirq.TParamVal'] = {'theta_default': 0.0, 'zeta_default': 0.0, 'chi_default': 0.0, 'gamma_default': 0.0, 'phi_default': 0.0}
    if gate == ops.SQRT_ISWAP:
        defaults['theta_default'] = -np.pi / 4
        return defaults
    if gate == ops.SQRT_ISWAP_INV:
        defaults['theta_default'] = np.pi / 4
        return defaults
    if isinstance(gate, ops.FSimGate):
        defaults['theta_default'] = gate.theta
        defaults['phi_default'] = gate.phi
        return defaults
    if isinstance(gate, ops.PhasedFSimGate):
        return {'theta_default': gate.theta, 'zeta_default': gate.zeta, 'chi_default': gate.chi, 'gamma_default': gate.gamma, 'phi_default': gate.phi}
    raise ValueError(f'Unknown default angles for {gate}.')