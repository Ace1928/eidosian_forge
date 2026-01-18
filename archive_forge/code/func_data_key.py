from __future__ import annotations
from abc import ABC
from enum import Enum
from typing import Any
import numpy as np
from qiskit.pulse.channels import Channel
from qiskit.visualization.pulse_v2 import types
from qiskit.visualization.exceptions import VisualizationError
@property
def data_key(self):
    """Return unique hash of this object."""
    return str(hash((self.__class__.__name__, self.data_type, tuple(self.xvals), tuple(self.yvals))))