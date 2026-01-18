import pytest
from flaky import flaky
from scipy.sparse import csr_matrix
import pennylane as qml
from pennylane import numpy as np
from pennylane.measurements import (
def process_samples(self, samples, wire_order, shot_range=None, bin_size=None):
    return 1