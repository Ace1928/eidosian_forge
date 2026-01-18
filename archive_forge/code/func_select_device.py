from contextlib import contextmanager
from .cudadrv.devices import require_context, reset, gpus  # noqa: F401
from .kernel import FakeCUDAKernel
from numba.core.sigutils import is_signature
from warnings import warn
from ..args import In, Out, InOut  # noqa: F401
def select_device(dev=0):
    assert dev == 0, 'Only a single device supported by the simulator'