import copy
from threading import RLock
import pennylane as qml
from pennylane.measurements import CountsMP, ProbabilityMP, SampleMP, MeasurementProcess
from pennylane.operation import DecompositionUndefinedError, Operator, StatePrepBase
from pennylane.queuing import AnnotatedQueue, QueuingManager, process_queue
from pennylane.pytrees import register_pytree
from .qscript import QuantumScript
def stop_at(obj):
    return False