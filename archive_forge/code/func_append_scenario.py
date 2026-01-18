import copy
import itertools
from collections import OrderedDict
from typing import Sequence, Callable
import numpy as np
import pennylane as qml
from pennylane.transforms import transform
from pennylane import adjoint
from pennylane.ops.qubit.attributes import symmetric_over_all_wires
from pennylane.tape import QuantumTape, QuantumScript
from pennylane.transforms.commutation_dag import commutation_dag
from pennylane.wires import Wires
def append_scenario(self, matching):
    """
        Append a scenario to the list.
        Args:
            matching (MatchingScenarios): a scenario of match.
        """
    self.matching_scenarios_list.append(matching)