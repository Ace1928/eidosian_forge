from fractions import Fraction
from typing import Dict, Any, List, Tuple
import numpy as np
from qiskit.pulse import instructions
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_raw_operand_values_compact(data: types.PulseInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate the formatted virtual Z rotation label and the frequency change label
    from provided frame instruction.

    Raw operand values are shown in compact form. Frequency change is expressed
    in scientific notation. Values are shown in two lines.

    For example:
        - A phase change 1.57 and frequency change 1,234,567 are written by `1.57\\n1.2e+06`

    Stylesheets:
        - The `frame_change` style is applied.
        - The `annotate` style is applied for font size.

    Args:
        data: Frame change instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    style = {'zorder': formatter['layer.frame_change'], 'color': formatter['color.frame_change'], 'size': formatter['text_size.annotate'], 'va': 'center', 'ha': 'center'}
    if data.frame.freq == 0:
        freq_sci_notation = '0.0'
    else:
        abs_freq = np.abs(data.frame.freq)
        freq_sci_notation = '{base:.1f}e{exp:d}'.format(base=data.frame.freq / 10 ** int(np.floor(np.log10(abs_freq))), exp=int(np.floor(np.log10(abs_freq))))
    frame_info = f'{data.frame.phase:.2f}\n{freq_sci_notation}'
    text = drawings.TextData(data_type=types.LabelType.FRAME, channels=data.inst[0].channel, xvals=[data.t0], yvals=[1.2 * formatter['label_offset.frame_change']], text=frame_info, ignore_scaling=True, styles=style)
    return [text]