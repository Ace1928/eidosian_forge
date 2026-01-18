from typing import Union, TYPE_CHECKING
import abc
from cirq import circuits, devices, ops
from cirq.contrib.acquaintance.gates import AcquaintanceOpportunityGate, SwapNetworkGate
from cirq.contrib.acquaintance.bipartite import BipartiteSwapNetworkGate
from cirq.contrib.acquaintance.shift_swap_network import ShiftSwapNetworkGate
from cirq.contrib.acquaintance.permutation import PermutationGate
def get_acquaintance_size(obj: Union[circuits.Circuit, ops.Operation]) -> int:
    """The maximum number of qubits to be acquainted with each other."""
    if isinstance(obj, circuits.Circuit):
        return max(tuple((get_acquaintance_size(op) for op in obj.all_operations())) or (0,))
    if not isinstance(obj, ops.Operation):
        raise TypeError('not isinstance(obj, (Circuit, Operation))')
    if not isinstance(obj, ops.GateOperation):
        return 0
    if isinstance(obj.gate, AcquaintanceOpportunityGate):
        return len(obj.qubits)
    if isinstance(obj.gate, BipartiteSwapNetworkGate):
        return 2
    if isinstance(obj.gate, ShiftSwapNetworkGate):
        return obj.gate.acquaintance_size()
    if isinstance(obj.gate, SwapNetworkGate):
        if obj.gate.acquaintance_size is None:
            return sum(sorted(obj.gate.part_lens)[-2:])
        if obj.gate.acquaintance_size - 1 in obj.gate.part_lens:
            return obj.gate.acquaintance_size
    sizer = getattr(obj.gate, '_acquaintance_size_', None)
    return 0 if sizer is None else sizer(len(obj.qubits))