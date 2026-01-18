from numbers import Number
from collections import namedtuple
import numpy as np
import rustworkx as rx
from pennylane.measurements import MeasurementProcess
from pennylane.resource import ResourcesOperation
def print_contents(self):
    """Prints the contents of the quantum circuit."""
    print('Operations')
    print('==========')
    for op in self.operations:
        print(repr(op))
    print('\nObservables')
    print('===========')
    for op in self.observables:
        print(repr(op))