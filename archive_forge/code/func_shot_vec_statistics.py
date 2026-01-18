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
def shot_vec_statistics(self, circuit: QuantumTape):
    """Process measurement results from circuit execution using a device
        with a shot vector and return statistics.

        This is an auxiliary method of execute and uses statistics.

        When using shot vectors, measurement results for each item of the shot
        vector are contained in a tuple.

        Args:
            circuit (~.tape.QuantumTape): circuit to execute on the device

        Raises:
            QuantumFunctionError: if the value of :attr:`~.Observable.return_type` is not supported

        Returns:
            tuple: statistics for each shot item from the shot vector
        """
    results = []
    s1 = 0
    measurements = circuit.measurements
    counts_exist = any((isinstance(m, CountsMP) for m in measurements))
    single_measurement = len(measurements) == 1
    for shot_tuple in self._shot_vector:
        s2 = s1 + np.prod(shot_tuple)
        r = self.statistics(circuit, shot_range=[s1, s2], bin_size=shot_tuple.shots)
        if single_measurement:
            r = r[0]
        elif shot_tuple.copies == 1:
            r = tuple((r_[0] if isinstance(r_, list) else r_.T for r_ in r))
        elif counts_exist:
            r = self._multi_meas_with_counts_shot_vec(circuit, shot_tuple, r)
        else:
            r = [tuple((self._asarray(r_.T[idx]) for r_ in r)) for idx in range(shot_tuple.copies)]
        if isinstance(r, qml.numpy.ndarray):
            if shot_tuple.copies > 1:
                results.extend([self._asarray(r_) for r_ in qml.math.unstack(r.T)])
            else:
                results.append(r.T)
        elif single_measurement and counts_exist:
            results.extend(r)
        elif not single_measurement and shot_tuple.copies > 1:
            r = [tuple((elem if isinstance(elem, dict) else elem.T for elem in r_)) for r_ in r]
            results.extend(r)
        else:
            results.append(r)
        s1 = s2
    return tuple(results)