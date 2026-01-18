import abc
import collections
import dataclasses
import functools
import math
import re
from typing import (
import numpy as np
import pandas as pd
import cirq
from cirq.experiments.xeb_fitting import XEBPhasedFSimCharacterizationOptions
from cirq_google.api import v2
from cirq_google.engine import (
from cirq_google.ops import FSimGateFamily, SycamoreGate
def parameters_for_qubits_swapped(self) -> 'PhasedFSimCharacterization':
    """Parameters for the gate with qubits swapped between each other.

        The angles theta, gamma and phi are kept unchanged. The angles zeta and chi are negated for
        the gate with swapped qubits.

        Returns:
            New instance with angles adjusted for swapped qubits.
        """
    return PhasedFSimCharacterization(theta=self.theta, zeta=-self.zeta if self.zeta is not None else None, chi=-self.chi if self.chi is not None else None, gamma=self.gamma, phi=self.phi)