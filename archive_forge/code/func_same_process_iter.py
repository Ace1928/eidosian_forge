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
def same_process_iter():
    for batch in self._batch_sampler:
        ret = self._batchify_fn([self._dataset[idx] for idx in batch])
        if self._pin_memory:
            ret = _as_in_context(ret, context.cpu_pinned(self._pin_device_id))
        yield ret