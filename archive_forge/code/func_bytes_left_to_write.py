import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def bytes_left_to_write(self):
    """Determine if there are bytes left to write.

        :returns: bool -- ``True`` if there are bytes left to write, otherwise
            ``False``
        """
    to_read = 0
    if self.headers_unread:
        to_read += len(self.headers)
    return to_read + total_len(self.body) > 0