from __future__ import absolute_import
import io
import time
@bytesize.setter
def bytesize(self, bytesize):
    """Change byte size."""
    if bytesize not in self.BYTESIZES:
        raise ValueError('Not a valid byte size: {!r}'.format(bytesize))
    self._bytesize = bytesize
    if self.is_open:
        self._reconfigure_port()