from __future__ import annotations
from collections.abc import Iterator, Sequence
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import chain
import numpy as np
from qiskit import pulse
from qiskit.pulse.transforms import target_qobj_transform
from qiskit.visualization.exceptions import VisualizationError
from qiskit.visualization.pulse_v2 import events, types, drawings, device_info
from qiskit.visualization.pulse_v2.stylesheet import QiskitPulseStyle
def set_time_range(self, t_start: float, t_end: float, seconds: bool=True):
    """Set time range to draw.

        All child chart instances are updated when time range is updated.

        Args:
            t_start: Left boundary of drawing in units of cycle time or real time.
            t_end: Right boundary of drawing in units of cycle time or real time.
            seconds: Set `True` if times are given in SI unit rather than dt.

        Raises:
            VisualizationError: When times are given in float without specifying dt.
        """
    if seconds:
        if self.device.dt is not None:
            t_start = int(np.round(t_start / self.device.dt))
            t_end = int(np.round(t_end / self.device.dt))
        else:
            raise VisualizationError('Setting time range with SI units requires backend `dt` information.')
    self.time_range = (t_start, t_end)