from copy import copy
from typing import Callable, Union, Sequence
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.operation import Operator
from pennylane.workflow import QNode
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumTape
def null_processing_fn(res):
    return res[0]