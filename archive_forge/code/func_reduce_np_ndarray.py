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
def reduce_np_ndarray(data):
    """Reduce ndarray to shared memory handle"""
    data = data.as_in_context(context.Context('cpu_shared', 0))
    pid, fd, shape, dtype = data._to_shared_mem()
    fd = multiprocessing.reduction.DupFd(fd)
    return (rebuild_np_ndarray, (pid, fd, shape, dtype))