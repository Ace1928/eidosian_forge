from typing import Dict, Any, List
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_chart_scale(data: types.ChartAxis, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate the current scaling value of the chart.

    Stylesheets:
        - The `axis_label` style is applied.
        - The `annotate` style is partially applied for the font size.

    Args:
        data: Chart axis data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    style = {'zorder': formatter['layer.axis_label'], 'color': formatter['color.axis_label'], 'size': formatter['text_size.annotate'], 'va': 'center', 'ha': 'right'}
    scale_val = f'x{types.DynamicString.SCALE}'
    text = drawings.TextData(data_type=types.LabelType.CH_INFO, channels=data.channels, xvals=[types.AbstractCoordinate.LEFT], yvals=[-formatter['label_offset.chart_info']], text=scale_val, ignore_scaling=True, styles=style)
    return [text]