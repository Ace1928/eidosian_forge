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
@property
def circuit_hash(self):
    """The hash of the circuit upon the last execution.

        This can be used by devices in :meth:`~.apply` for parametric compilation.
        """
    raise NotImplementedError