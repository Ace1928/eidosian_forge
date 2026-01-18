import numpy as np
import pennylane as qml
from pennylane.pauli.utils import is_pauli_word, pauli_to_binary, _wire_map_from_pauli_pair
def is_commuting(operation1, operation2, wire_map=None):
    """Check if two operations are commuting using a lookup table.

    A lookup table is used to check the commutation between the
    controlled, targeted part of operation 1 with the controlled, targeted part of operation 2.

    .. note::

        Most qubit-based PennyLane operations are supported --- CV operations
        are not supported at this time.

        Unsupported qubit-based operations include:

        :class:`~.PauliRot`, :class:`~.QubitDensityMatrix`, :class:`~.CVNeuralNetLayers`,
        :class:`~.ApproxTimeEvolution`, :class:`~.ArbitraryUnitary`, :class:`~.CommutingEvolution`,
        :class:`~.DisplacementEmbedding`, :class:`~.SqueezingEmbedding`, :class:`~.Prod`,
        :class:`~.Sum`, :class:`~.Exp` and :class:`~.SProd`.

    Args:
        operation1 (.Operation): A first quantum operation.
        operation2 (.Operation): A second quantum operation.
        wire_map (dict[Union[str, int], int]): dictionary for Pauli word commutation containing all
            wire labels used in the Pauli word as keys, and unique integer labels as their values

    Returns:
         bool: True if the operations commute, False otherwise.

    **Example**

    >>> qml.is_commuting(qml.X(0), qml.Z(0))
    False
    """
    if operation1.name in unsupported_operations or isinstance(operation1, (qml.operation.CVOperation, qml.operation.Channel)):
        raise qml.QuantumFunctionError(f'Operation {operation1.name} not supported.')
    if operation2.name in unsupported_operations or isinstance(operation2, (qml.operation.CVOperation, qml.operation.Channel)):
        raise qml.QuantumFunctionError(f'Operation {operation2.name} not supported.')
    if is_pauli_word(operation1) and is_pauli_word(operation2):
        return _pword_is_commuting(operation1, operation2, wire_map)
    if isinstance(operation1, qml.operation.Tensor) or isinstance(operation2, qml.operation.Tensor):
        raise qml.QuantumFunctionError('Tensor operations are not supported.')
    if not intersection(operation1.wires, operation2.wires):
        return True
    with qml.QueuingManager.stop_recording():
        operation1 = qml.simplify(operation1)
        operation2 = qml.simplify(operation2)
    if operation1.name in non_commuting_operations or operation2.name in non_commuting_operations:
        return False
    if operation1.name == 'CRot' and operation2.name == 'CRot':
        return check_commutation_two_non_simplified_crot(operation1, operation2)
    if 'Identity' in (operation1.name, operation2.name):
        return True
    op_set = {'U2', 'U3', 'Rot', 'CRot'}
    if operation1.name in op_set and operation2.name in op_set:
        return check_commutation_two_non_simplified_rotations(operation1, operation2)
    ctrl_base_1 = _get_target_name(operation1)
    ctrl_base_2 = _get_target_name(operation2)
    op1_control_wires = getattr(operation1, 'control_wires', {})
    op2_control_wires = getattr(operation2, 'control_wires', {})
    target_wires_1 = qml.wires.Wires([w for w in operation1.wires if w not in op1_control_wires])
    target_wires_2 = qml.wires.Wires([w for w in operation2.wires if w not in op2_control_wires])
    if intersection(target_wires_1, target_wires_2) and (not _commutes(ctrl_base_1, ctrl_base_2)):
        return False
    if intersection(target_wires_1, op2_control_wires) and (not _commutes('ctrl', ctrl_base_1)):
        return False
    if intersection(target_wires_2, op1_control_wires) and (not _commutes('ctrl', ctrl_base_2)):
        return False
    return True