import contextlib
from datetime import datetime
import sys
import time
@contextlib.contextmanager
def scalars_tracker(self, num_scalars):
    """Create a context manager for tracking a scalar batch upload.

        Args:
          num_scalars: Number of scalars in the batch.
        """
    self._overwrite_line_message('Uploading %d scalars' % num_scalars)
    try:
        yield
    finally:
        self._stats.add_scalars(num_scalars)