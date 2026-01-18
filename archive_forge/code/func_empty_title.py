from collections import defaultdict
from typing import List, Dict, Any, Tuple, Iterator, Optional, Union
import numpy as np
from qiskit import pulse
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.pulse_v2.device_info import DrawerBackendInfo
def empty_title(program: Union[pulse.Waveform, pulse.Schedule], device: DrawerBackendInfo) -> str:
    """Layout function for generating an empty figure title."""
    return ''