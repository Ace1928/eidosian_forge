import dataclasses
from abc import abstractmethod, ABC
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd
import sympy
from cirq import circuits, ops, protocols, _import
from cirq.experiments.xeb_simulation import simulate_2q_xeb_circuits
def get_parameterized_gate(self):
    theta = THETA_SYMBOL if self.characterize_theta else self.theta_default
    zeta = ZETA_SYMBOL if self.characterize_zeta else self.zeta_default
    chi = CHI_SYMBOL if self.characterize_chi else self.chi_default
    gamma = GAMMA_SYMBOL if self.characterize_gamma else self.gamma_default
    phi = PHI_SYMBOL if self.characterize_phi else self.phi_default
    return ops.PhasedFSimGate(theta=theta, zeta=zeta, chi=chi, gamma=gamma, phi=phi)