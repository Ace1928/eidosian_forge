from numbers import Number
from collections import namedtuple
import numpy as np
import rustworkx as rx
from pennylane.measurements import MeasurementProcess
from pennylane.resource import ResourcesOperation
@property
def max_simultaneous_measurements(self):
    """Returns the maximum number of measurements on any wire in the circuit graph.

        This method counts the number of measurements for each wire and returns
        the maximum.

        **Examples**


        >>> dev = qml.device('default.qubit', wires=3)
        >>> def circuit_measure_max_once():
        ...     return qml.expval(qml.X(0))
        >>> qnode = qml.QNode(circuit_measure_max_once, dev)
        >>> qnode()
        >>> qnode.qtape.graph.max_simultaneous_measurements
        1
        >>> def circuit_measure_max_twice():
        ...     return qml.expval(qml.X(0)), qml.probs(wires=0)
        >>> qnode = qml.QNode(circuit_measure_max_twice, dev)
        >>> qnode()
        >>> qnode.qtape.graph.max_simultaneous_measurements
        2

        Returns:
            int: the maximum number of measurements
        """
    if self._max_simultaneous_measurements is None:
        all_wires = []
        for obs in self.observables:
            all_wires.extend(obs.wires.tolist())
        a = np.array(all_wires)
        _, counts = np.unique(a, return_counts=True)
        self._max_simultaneous_measurements = counts.max() if counts.size != 0 else 1
    return self._max_simultaneous_measurements