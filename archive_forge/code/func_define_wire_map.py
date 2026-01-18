import functools
import numpy as np
import pennylane as qml  # pylint: disable=unused-import
from pennylane import QutritDevice, QutritBasisState, DeviceError
from pennylane.wires import WireError
from pennylane.devices.default_qubit_legacy import _get_slice
from .._version import __version__
def define_wire_map(self, wires):
    consecutive_wires = range(self.num_wires)
    wire_map = zip(wires, consecutive_wires)
    return dict(wire_map)