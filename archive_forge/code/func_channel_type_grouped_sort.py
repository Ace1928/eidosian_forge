from collections import defaultdict
from typing import List, Dict, Any, Tuple, Iterator, Optional, Union
import numpy as np
from qiskit import pulse
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.pulse_v2.device_info import DrawerBackendInfo
def channel_type_grouped_sort(channels: List[pulse.channels.Channel], formatter: Dict[str, Any], device: DrawerBackendInfo) -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Layout function for the channel assignment to the chart instance.

    Assign single channel per chart. Channels are grouped by type and
    sorted by index in ascending order.

    Stylesheet key:
        `chart_channel_map`

    For example:
        [D0, D2, C0, C2, M0, M2, A0, A2] -> [D0, D2, C0, C2, M0, M2, A0, A2]

    Args:
        channels: Channels to show.
        formatter: Dictionary of stylesheet settings.
        device: Backend configuration.

    Yields:
        Tuple of chart name and associated channels.
    """
    chan_type_dict = defaultdict(list)
    for chan in channels:
        chan_type_dict[type(chan)].append(chan)
    ordered_channels = []
    d_chans = chan_type_dict.get(pulse.DriveChannel, [])
    ordered_channels.extend(sorted(d_chans, key=lambda x: x.index))
    c_chans = chan_type_dict.get(pulse.ControlChannel, [])
    ordered_channels.extend(sorted(c_chans, key=lambda x: x.index))
    m_chans = chan_type_dict.get(pulse.MeasureChannel, [])
    ordered_channels.extend(sorted(m_chans, key=lambda x: x.index))
    if formatter['control.show_acquire_channel']:
        a_chans = chan_type_dict.get(pulse.AcquireChannel, [])
        ordered_channels.extend(sorted(a_chans, key=lambda x: x.index))
    for chan in ordered_channels:
        yield (chan.name.upper(), [chan])