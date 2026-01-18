from typing import Dict, Any, List
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_channel_freqs(data: types.ChartAxis, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate the frequency values of associated channels.

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
    if len(data.channels) > 1:
        sources = []
        for chan in data.channels:
            freq = device.get_channel_frequency(chan)
            if not freq:
                continue
            sources.append(f'{chan.name.upper()}: {freq / 1000000000.0:.2f} GHz')
        freq_text = ', '.join(sources)
    else:
        freq = device.get_channel_frequency(data.channels[0])
        if freq:
            freq_text = f'{freq / 1000000000.0:.2f} GHz'
        else:
            freq_text = ''
    text = drawings.TextData(data_type=types.LabelType.CH_INFO, channels=data.channels, xvals=[types.AbstractCoordinate.LEFT], yvals=[-formatter['label_offset.chart_info']], text=freq_text or 'no freq.', ignore_scaling=True, styles=style)
    return [text]