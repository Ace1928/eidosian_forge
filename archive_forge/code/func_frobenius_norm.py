import numpy
from .. import registry
from ..compat import cublas, cupy, cupyx
from ..types import DeviceTypes
from ..util import (
from . import _custom_kernels
from .numpy_ops import NumpyOps
from .ops import Ops
def frobenius_norm(X):
    X_vec = X.reshape(-1)
    return cublas.nrm2(X_vec)