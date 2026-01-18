import contextlib
import io
import os
from uuid import uuid4
import requests
from .._compat import fields
def smart_truncate(self):
    to_be_read = total_len(self)
    already_read = self._get_end() - to_be_read
    if already_read >= to_be_read:
        old_bytes = self.read()
        self.seek(0, 0)
        self.truncate()
        self.write(old_bytes)
        self.seek(0, 0)