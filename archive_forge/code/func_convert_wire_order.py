from pennylane.ops import Controlled, Conditional
from pennylane.measurements import MeasurementProcess, MidMeasureMP, MeasurementValue
def convert_wire_order(tape, wire_order=None, show_all_wires=False):
    """Creates the mapping between wire labels and place in order.

    Args:
        tape (~.tape.QuantumTape): the Quantum Tape containing operations and measurements
        wire_order Sequence[Any]: the order (from top to bottom) to print the wires

    Keyword Args:
        show_all_wires=False (bool): whether to display all wires in ``wire_order``
            or only include ones used by operations in ``ops``

    Returns:
        dict: map from wire labels to sequential positive integers
    """
    default = default_wire_map(tape)
    if wire_order is None:
        return default
    wire_order = list(wire_order) + [wire for wire in default if wire not in wire_order]
    if not show_all_wires:
        used_wires = {wire for op in tape for wire in op.wires}
        wire_order = [wire for wire in wire_order if wire in used_wires]
    return {wire: ind for ind, wire in enumerate(wire_order)}