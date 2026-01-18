import warnings
from typing import List, Union, Dict, Any, Optional
from qiskit.circuit import Qubit, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.visualization.timeline import types, drawings
def gen_timeslot(bit: types.Bits, formatter: Dict[str, Any]) -> List[drawings.BoxData]:
    """Generate time slot of associated bit.

    Stylesheet:
        - `timeslot` style is applied.

    Args:
        bit: Bit object associated to this drawing.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `BoxData` drawings.
    """
    styles = {'zorder': formatter['layer.timeslot'], 'alpha': formatter['alpha.timeslot'], 'linewidth': formatter['line_width.timeslot'], 'facecolor': formatter['color.timeslot']}
    drawing = drawings.BoxData(data_type=types.BoxType.TIMELINE, xvals=[types.AbstractCoordinate.LEFT, types.AbstractCoordinate.RIGHT], yvals=[-0.5 * formatter['box_height.timeslot'], 0.5 * formatter['box_height.timeslot']], bit=bit, styles=styles)
    return [drawing]