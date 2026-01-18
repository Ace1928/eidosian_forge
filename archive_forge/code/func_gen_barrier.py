from typing import Dict, Any, List
from qiskit.visualization.pulse_v2 import drawings, types, device_info
def gen_barrier(data: types.BarrierInstruction, formatter: Dict[str, Any], device: device_info.DrawerBackendInfo) -> List[drawings.LineData]:
    """Generate the barrier from provided relative barrier instruction.

    Stylesheets:
        - The `barrier` style is applied.

    Args:
        data: Barrier instruction data to draw.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.
    Returns:
        List of `LineData` drawings.
    """
    style = {'alpha': formatter['alpha.barrier'], 'zorder': formatter['layer.barrier'], 'linewidth': formatter['line_width.barrier'], 'linestyle': formatter['line_style.barrier'], 'color': formatter['color.barrier']}
    line = drawings.LineData(data_type=types.LineType.BARRIER, channels=data.channels, xvals=[data.t0, data.t0], yvals=[types.AbstractCoordinate.BOTTOM, types.AbstractCoordinate.TOP], styles=style)
    return [line]