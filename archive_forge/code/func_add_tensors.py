import contextlib
from datetime import datetime
import sys
import time
def add_tensors(self, num_tensors, num_tensors_skipped, tensor_bytes, tensor_bytes_skipped):
    """Add a batch of tensors.

        Args:
          num_tensors: Number of tensors encountered in this batch, including
            the ones skipped due to reasons such as large exceeding limit.
          num_tensors: Number of tensors skipped. This describes a subset of
            `num_tensors` and hence must be `<= num_tensors`.
          tensor_bytes: Total byte size of tensors encountered in this batch,
            including the skipped ones.
          tensor_bytes_skipped: Total byte size of the tensors skipped due to
            reasons such as size exceeding limit.
        """
    assert num_tensors_skipped <= num_tensors
    assert tensor_bytes_skipped <= tensor_bytes
    self._refresh_last_data_added_timestamp()
    self._num_tensors += num_tensors
    self._num_tensors_skipped += num_tensors_skipped
    self._tensor_bytes += tensor_bytes
    self._tensor_bytes_skipped = tensor_bytes_skipped