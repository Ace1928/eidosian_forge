from typing import Dict, Any, List
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_snapshot_symbol(data: types.SnapshotInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.TextData]:
    """Generate a snapshot symbol with instruction meta data from provided snapshot instruction.

    Stylesheets:
        - The `snapshot` style is applied for snapshot symbol.
        - The symbol type in unicode is specified in `formatter.unicode_symbol.snapshot`.
        - The symbol type in latex is specified in `formatter.latex_symbol.snapshot`.

    Args:
        data: Snapshot instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Returns:
        List of `TextData` drawings.
    """
    style = {'zorder': formatter['layer.snapshot'], 'color': formatter['color.snapshot'], 'size': formatter['text_size.snapshot'], 'va': 'bottom', 'ha': 'center'}
    meta = {'snapshot type': data.inst.type, 't0 (cycle time)': data.t0, 't0 (sec)': data.t0 * data.dt if data.dt else 'N/A', 'name': data.inst.name, 'label': data.inst.label}
    text = drawings.TextData(data_type=types.SymbolType.SNAPSHOT, channels=data.inst.channel, xvals=[data.t0], yvals=[0], text=formatter['unicode_symbol.snapshot'], latex=formatter['latex_symbol.snapshot'], ignore_scaling=True, meta=meta, styles=style)
    return [text]