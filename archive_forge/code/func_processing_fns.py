from itertools import islice, product
from typing import List
import numpy as np
import pennylane as qml
from pennylane import BasisState, QubitDevice, StatePrep
from pennylane.devices import DefaultQubitLegacy
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operation
from pennylane.wires import Wires
from ._serialize import QuantumScriptSerializer
from ._version import __version__
def processing_fns(tapes):
    vjps = []
    for tape, fun in zip(tapes, fns):
        vjp = fun(tape)
        if not isinstance(vjp, tuple) and getattr(reduction, '__name__', reduction) == 'extend':
            vjp = (vjp,)
        if isinstance(reduction, str):
            getattr(vjps, reduction)(vjp)
        elif callable(reduction):
            reduction(vjps, vjp)
    return vjps