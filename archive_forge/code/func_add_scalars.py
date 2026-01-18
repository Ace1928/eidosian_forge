import contextlib
from datetime import datetime
import sys
import time
def add_scalars(self, num_scalars):
    """Add a batch of scalars.

        Args:
          num_scalars: Number of scalars uploaded in this batch.
        """
    self._refresh_last_data_added_timestamp()
    self._num_scalars += num_scalars