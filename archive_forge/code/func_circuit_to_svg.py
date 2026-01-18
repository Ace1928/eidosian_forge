from typing import TYPE_CHECKING, List, Tuple, cast, Dict
import matplotlib.textpath
import matplotlib.font_manager
def circuit_to_svg(circuit: 'cirq.Circuit') -> str:
    """Render a circuit as SVG."""
    _validate_circuit(circuit)
    tdd = circuit.to_text_diagram_drawer(transpose=False)
    if len(tdd.horizontal_lines) == 0:
        return '<svg></svg>'
    return tdd_to_svg(tdd)