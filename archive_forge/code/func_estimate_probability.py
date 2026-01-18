import abc
import itertools
import warnings
from collections import defaultdict
from typing import Union, List
import inspect
import logging
import numpy as np
import pennylane as qml
from pennylane import Device, DeviceError
from pennylane.math import multiply as qmlmul
from pennylane.math import sum as qmlsum
from pennylane.measurements import (
from pennylane.resource import Resources
from pennylane.operation import operation_derivative, Operation
from pennylane.tape import QuantumTape
from pennylane.wires import Wires
def estimate_probability(self, wires=None, shot_range=None, bin_size=None):
    """Return the estimated probability of each computational basis state
        using the generated samples.

        Args:
            wires (Iterable[Number, str], Number, str, Wires): wires to calculate
                marginal probabilities for. Wires not provided are traced out of the system.
            shot_range (tuple[int]): 2-tuple of integers specifying the range of samples
                to use. If not specified, all samples are used.
            bin_size (int): Divides the shot range into bins of size ``bin_size``, and
                returns the measurement statistic separately over each bin. If not
                provided, the entire shot range is treated as a single bin.

        Returns:
            array[float]: list of the probabilities
        """
    wires = wires or self.wires
    wires = Wires(wires)
    device_wires = self.map_wires(wires)
    num_wires = len(device_wires)
    if shot_range is None:
        samples = self._samples[..., device_wires]
    else:
        samples = self._samples[..., slice(*shot_range), device_wires]
    powers_of_two = 2 ** np.arange(num_wires)[::-1]
    indices = samples @ powers_of_two
    batch_size = self._samples.shape[0] if np.ndim(self._samples) == 3 else None
    dim = 2 ** num_wires
    if bin_size is not None:
        num_bins = samples.shape[-2] // bin_size
        prob = self._count_binned_samples(indices, batch_size, dim, bin_size, num_bins)
    else:
        prob = self._count_unbinned_samples(indices, batch_size, dim)
    return self._asarray(prob, dtype=self.R_DTYPE)