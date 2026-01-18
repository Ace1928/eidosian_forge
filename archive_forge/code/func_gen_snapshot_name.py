from typing import Dict, Any, List
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_snapshot_name(data: types.SnapshotInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate the name of snapshot.

    Stylesheets:
        - The `snapshot` style is applied for snapshot symbol.
        - The `annotate` style is applied for label font size.

    Args:
        data: Snapshot instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    style = {'zorder': formatter['layer.snapshot'], 'color': formatter['color.snapshot'], 'size': formatter['text_size.annotate'], 'va': 'center', 'ha': 'center'}
    text = drawings.TextData(data_type=types.LabelType.SNAPSHOT, channels=data.inst.channel, xvals=[data.t0], yvals=[formatter['label_offset.snapshot']], text=data.inst.name, ignore_scaling=True, styles=style)
    return [text]