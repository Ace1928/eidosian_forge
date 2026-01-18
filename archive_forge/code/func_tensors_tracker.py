import contextlib
from datetime import datetime
import sys
import time
@contextlib.contextmanager
def tensors_tracker(self, num_tensors, num_tensors_skipped, tensor_bytes, tensor_bytes_skipped):
    """Create a context manager for tracking a tensor batch upload.

        Args:
          num_tensors: Total number of tensors in the batch.
          num_tensors_skipped: Number of tensors skipped (a subset of
            `num_tensors`). Hence this must be `<= num_tensors`.
          tensor_bytes: Total byte size of the tensors in the batch.
          tensor_bytes_skipped: Byte size of skipped tensors in the batch (a
            subset of `tensor_bytes`). Must be `<= tensor_bytes`.
        """
    if num_tensors_skipped:
        message = 'Uploading %d tensors (%s) (Skipping %d tensors, %s)' % (num_tensors - num_tensors_skipped, readable_bytes_string(tensor_bytes - tensor_bytes_skipped), num_tensors_skipped, readable_bytes_string(tensor_bytes_skipped))
    else:
        message = 'Uploading %d tensors (%s)' % (num_tensors, readable_bytes_string(tensor_bytes))
    self._overwrite_line_message(message)
    try:
        yield
    finally:
        self._stats.add_tensors(num_tensors, num_tensors_skipped, tensor_bytes, tensor_bytes_skipped)