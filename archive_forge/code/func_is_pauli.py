import numpy as np
import pytest
import matplotlib.pyplot as plt
import cirq
import cirq.experiments.qubit_characterizations as ceqc
from cirq import GridQubit
from cirq import circuits, ops, sim
from cirq.experiments import (
def is_pauli(u):
    return any((cirq.equal_up_to_global_phase(u, p) for p in PAULIS))