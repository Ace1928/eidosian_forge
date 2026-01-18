import pickle
import io
import sys
import signal
import multiprocessing
import multiprocessing.queues
from multiprocessing.reduction import ForkingPickler
from multiprocessing.pool import ThreadPool
import threading
import numpy as np
from . import sampler as _sampler
from ... import nd, context
from ...util import is_np_shape, is_np_array, set_np
from ... import numpy as _mx_np  # pylint: disable=reimported
def rebuild_ndarray(pid, fd, shape, dtype):
    """Rebuild ndarray from pickled shared memory"""
    fd = fd.detach()
    return nd.NDArray(nd.ndarray._new_from_shared_mem(pid, fd, shape, dtype))