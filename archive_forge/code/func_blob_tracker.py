import contextlib
from datetime import datetime
import sys
import time
@contextlib.contextmanager
def blob_tracker(self, blob_bytes):
    """Creates context manager tracker for uploading a blob.

        Args:
          blob_bytes: Total byte size of the blob being uploaded.
        """
    self._overwrite_line_message('Uploading binary object (%s)' % readable_bytes_string(blob_bytes))
    try:
        yield _BlobTracker(self._stats, blob_bytes)
    finally:
        pass