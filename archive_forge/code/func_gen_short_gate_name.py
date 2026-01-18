import warnings
from typing import List, Union, Dict, Any, Optional
from qiskit.circuit import Qubit, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.visualization.timeline import types, drawings
def gen_short_gate_name(gate: types.ScheduledGate, formatter: Dict[str, Any]) -> List[drawings.TextData]:
    """Generate gate name.

    Only operand name is shown.

    Stylesheet:
        - `gate_name` style is applied.
        - `gate_latex_repr` key is used to find the latex representation of the gate name.

    Args:
        gate: Gate information source.
        formatter: Dictionary of stylesheet settings.

    Returns:
        List of `TextData` drawings.
    """
    if gate.duration > 0:
        v_align = 'center'
        v_pos = 0
    else:
        v_align = 'bottom'
        v_pos = formatter['label_offset.frame_change']
    styles = {'zorder': formatter['layer.gate_name'], 'color': formatter['color.gate_name'], 'size': formatter['text_size.gate_name'], 'va': v_align, 'ha': 'center'}
    default_name = f'{{\\rm {gate.operand.name}}}'
    latex_name = formatter['latex_symbol.gates'].get(gate.operand.name, default_name)
    label_plain = f'{gate.operand.name}'
    label_latex = f'{latex_name}'
    if gate.operand.name == 'delay':
        data_type = types.LabelType.DELAY
    else:
        data_type = types.LabelType.GATE_NAME
    drawing = drawings.TextData(data_type=data_type, xval=gate.t0 + 0.5 * gate.duration, yval=v_pos, bit=gate.bits[gate.bit_position], text=label_plain, latex=label_latex, styles=styles)
    return [drawing]