from contextlib import contextmanager
from ipywidgets import register
from traitlets import Unicode, Set, Undefined, Int, validate
import numpy as np
from ..widgets import DataWidget
from .traits import NDArray
from .serializers import compressed_array_serialization
from inspect import Signature, Parameter
def sync_segment(self, segments):
    """Sync a segments of contiguous memory.

        By only syncing a segment of the array, a full transmission of the
        updated array is avoided. However, this does put the responsibility
        of ensuring the correct sync state on the caller.

        This respects hold_sync, so several segments can be stacked with
        multiple calls when holding the sync.

        Parameters
        ----------
        segments : iterable of two-tuples
            An iterable collection of segments represented by (start, stop) tuples.
        """
    if self._holding_sync:
        self._segments_to_send.add(*segments)
    else:
        self.send_segment(segments)