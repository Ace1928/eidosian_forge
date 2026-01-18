from contextlib import contextmanager
from ipywidgets import register
from traitlets import Unicode, Set, Undefined, Int, validate
import numpy as np
from ..widgets import DataWidget
from .traits import NDArray
from .serializers import compressed_array_serialization
from inspect import Signature, Parameter
def send_segment(self, segments):
    """Send segments to the front-end.

        Note: This does not respect hold_sync. If that is wanted, use
        sync_segment instead.

        Parameters
        ----------
        segments : iterable of two-tuples
            An iterable collection of segments represented by (start, stop) tuples.
        """
    starts = []
    buffers = []
    raveled = np.ravel(self.array, order='C')
    length = len(raveled)
    for s in segments:
        starts.append(s[0] if s[0] >= 0 else length - s[0])
        buffers.append(np.ascontiguousarray(raveled[s[0]:s[1]]))
    msg = {'method': 'update_array_segment', 'name': 'array', 'starts': starts}
    self._send(msg, buffers)