from numbers import Number
from collections import namedtuple
import numpy as np
import rustworkx as rx
from pennylane.measurements import MeasurementProcess
from pennylane.resource import ResourcesOperation
def iterate_parametrized_layers(self):
    """Parametrized layers of the circuit.

        Returns:
            Iterable[LayerData]: layers with extra metadata
        """
    for ops, param_inds in self.parametrized_layers:
        pre_queue = self.ancestors_in_order(ops)
        post_queue = self.descendants_in_order(ops)
        yield LayerData(pre_queue, ops, tuple(param_inds), post_queue)