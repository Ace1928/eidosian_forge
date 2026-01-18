import hashlib
import os
import tempfile
def open_object(self, sha):
    """Open an object by sha."""
    try:
        return open(self._sha_path(sha), 'rb')
    except FileNotFoundError as exc:
        raise KeyError(sha) from exc