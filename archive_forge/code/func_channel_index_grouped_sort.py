from collections import defaultdict
from typing import List, Dict, Any, Tuple, Iterator, Optional, Union
import numpy as np
from qiskit import pulse
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.pulse_v2.device_info import DrawerBackendInfo
def channel_index_grouped_sort(channels: List[pulse.channels.Channel], formatter: Dict[str, Any], device: DrawerBackendInfo) -> Iterator[Tuple[str, List[pulse.channels.Channel]]]:
    """Layout function for the channel assignment to the chart instance.

    Assign single channel per chart. Channels are grouped by the same index and
    sorted by type.

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
    inds = set()
    for chan in channels:
        chan_type_dict[type(chan)].append(chan)
        inds.add(chan.index)
    d_chans = chan_type_dict.get(pulse.DriveChannel, [])
    d_chans = sorted(d_chans, key=lambda x: x.index, reverse=True)
    u_chans = chan_type_dict.get(pulse.ControlChannel, [])
    u_chans = sorted(u_chans, key=lambda x: x.index, reverse=True)
    m_chans = chan_type_dict.get(pulse.MeasureChannel, [])
    m_chans = sorted(m_chans, key=lambda x: x.index, reverse=True)
    a_chans = chan_type_dict.get(pulse.AcquireChannel, [])
    a_chans = sorted(a_chans, key=lambda x: x.index, reverse=True)
    ordered_channels = []
    for ind in sorted(inds):
        if len(d_chans) > 0 and d_chans[-1].index == ind:
            ordered_channels.append(d_chans.pop())
        if len(u_chans) > 0 and u_chans[-1].index == ind:
            ordered_channels.append(u_chans.pop())
        if len(m_chans) > 0 and m_chans[-1].index == ind:
            ordered_channels.append(m_chans.pop())
        if formatter['control.show_acquire_channel']:
            if len(a_chans) > 0 and a_chans[-1].index == ind:
                ordered_channels.append(a_chans.pop())
    for chan in ordered_channels:
        yield (chan.name.upper(), [chan])