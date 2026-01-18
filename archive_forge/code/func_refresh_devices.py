from importlib import reload, metadata
from sys import version_info
import numpy as _np
from semantic_version import SimpleSpec, Version
from pennylane.boolean_fn import BooleanFn
import pennylane.numpy
from pennylane.queuing import QueuingManager, apply
import pennylane.kernels
import pennylane.math
import pennylane.operation
import pennylane.qnn
import pennylane.templates
import pennylane.pauli
from pennylane.pauli import pauli_decompose
from pennylane.resource import specs
import pennylane.resource
import pennylane.qchem
from pennylane.fermi import FermiC, FermiA, jordan_wigner
from pennylane.qchem import (
from pennylane._device import Device, DeviceError
from pennylane._grad import grad, jacobian, vjp, jvp
from pennylane._qubit_device import QubitDevice
from pennylane._qutrit_device import QutritDevice
from pennylane._version import __version__
from pennylane.about import about
from pennylane.circuit_graph import CircuitGraph
from pennylane.configuration import Configuration
from pennylane.drawer import draw, draw_mpl
from pennylane.tracker import Tracker
from pennylane.io import *
from pennylane.measurements import (
from pennylane.ops import *
from pennylane.ops import adjoint, ctrl, cond, exp, sum, pow, prod, s_prod
from pennylane.templates import broadcast, layer
from pennylane.templates.embeddings import *
from pennylane.templates.layers import *
from pennylane.templates.tensornetworks import *
from pennylane.templates.swapnetworks import *
from pennylane.templates.state_preparations import *
from pennylane.templates.subroutines import *
from pennylane import qaoa
from pennylane.workflow import QNode, qnode, execute
from pennylane.transforms import (
from pennylane.ops.functions import (
from pennylane.ops.identity import I
from pennylane.optimize import *
from pennylane.debugging import snapshots
from pennylane.shadows import ClassicalShadow
from pennylane.qcut import cut_circuit, cut_circuit_mc
import pennylane.pulse
import pennylane.fourier
from pennylane.gradients import metric_tensor, adjoint_metric_tensor
import pennylane.gradients  # pylint:disable=wrong-import-order
import pennylane.qinfo
import pennylane.logging  # pylint:disable=wrong-import-order
from pennylane.compiler import qjit, while_loop, for_loop
import pennylane.compiler
import pennylane.data
import pennylane.interfaces
def refresh_devices():
    """Scan installed PennyLane plugins to refresh the device list."""
    global plugin_devices
    reload(metadata)
    plugin_devices = _get_device_entrypoints()