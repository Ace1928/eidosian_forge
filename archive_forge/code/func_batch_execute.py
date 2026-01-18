from collections import defaultdict
import numpy as np
from pennylane.ops.qubit.attributes import diagonal_in_z_basis
from pennylane import QubitDevice
from pennylane.measurements import Shots
from pennylane.resource import Resources
from .._version import __version__
def batch_execute(self, circuits, **kwargs):
    res = []
    for c in circuits:
        res.append(self.execute(c))
    if self.tracker.active:
        self.tracker.update(batches=1, batch_len=len(circuits))
        self.tracker.record()
    return res