import warnings
from typing import List, Union, Dict, Any, Optional
from qiskit.circuit import Qubit, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.visualization.timeline import types, drawings
def gen_gate_link(link: types.GateLink, formatter: Dict[str, Any]) -> List[drawings.GateLinkData]:
    """Generate gate link line.

    Line color depends on the operand type.

    Stylesheet:
        - `gate_link` style is applied.
        - The `gate_face_color` style is applied for line color.

    Args:
        link: Gate link object.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `GateLinkData` drawings.
    """
    color = formatter['color.gates'].get(link.opname, formatter['color.default_gate'])
    styles = {'alpha': formatter['alpha.gate_link'], 'zorder': formatter['layer.gate_link'], 'linewidth': formatter['line_width.gate_link'], 'linestyle': formatter['line_style.gate_link'], 'color': color}
    drawing = drawings.GateLinkData(bits=link.bits, xval=link.t0, styles=styles)
    return [drawing]