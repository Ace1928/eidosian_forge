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
def set_disable_type(self, data_type: types.DataTypes, remove: bool=True):
    """Interface method to control visibility of data types.

        Specified object in the blocked list will not be shown.

        Args:
            data_type: A drawing data type to disable.
            remove: Set `True` to disable, set `False` to enable.
        """
    if isinstance(data_type, Enum):
        data_type_str = str(data_type.value)
    else:
        data_type_str = data_type
    if remove:
        self.disable_types.add(data_type_str)
    else:
        self.disable_types.discard(data_type_str)