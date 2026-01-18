import abc
import copy
import types
import warnings
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from functools import lru_cache
import numpy as np
import pennylane as qml
from pennylane.measurements import (
from pennylane.operation import Observable, Operation, Tensor, Operator, StatePrepBase
from pennylane.ops import Hamiltonian, Sum
from pennylane.tape import QuantumScript, QuantumTape, expand_tape_state_prep
from pennylane.wires import WireError, Wires
from pennylane.queuing import QueuingManager
def hamiltonian_fn(res):
    return res[0]