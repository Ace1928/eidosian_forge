import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.operation import AnyWires, Operation, Operator
@property
def estimation_wires(self):
    """The estimation wires of the QPE"""
    return self._hyperparameters['estimation_wires']