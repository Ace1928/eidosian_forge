from numbers import Number
from collections import namedtuple
import numpy as np
import rustworkx as rx
from pennylane.measurements import MeasurementProcess
from pennylane.resource import ResourcesOperation
@property
def parametrized_layers(self):
    """Identify the parametrized layer structure of the circuit.

        Returns:
            list[Layer]: layers of the circuit
        """
    current = Layer([], [])
    layers = [current]
    for idx, info in enumerate(self.par_info):
        if idx in self.trainable_params:
            op = info['op']
            sub = self.ancestors((op,))
            if any((o1 is o2 for o1 in current.ops for o2 in sub)):
                current = Layer([], [])
                layers.append(current)
            current.ops.append(op)
            current.param_inds.append(idx)
    return layers