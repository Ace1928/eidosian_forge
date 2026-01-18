from typing import Dict, Optional, Tuple, TYPE_CHECKING
from cirq import circuits, ops
def swap_print_moment() -> 'cirq.Operation':
    return _SwapPrintGate(tuple(zip(qdict.values(), [inverse_map[x] for x in qdict.values()]))).on(*all_qubits)