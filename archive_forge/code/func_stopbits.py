from __future__ import absolute_import
import io
import time
@stopbits.setter
def stopbits(self, stopbits):
    """Change stop bits size."""
    if stopbits not in self.STOPBITS:
        raise ValueError('Not a valid stop bit size: {!r}'.format(stopbits))
    self._stopbits = stopbits
    if self.is_open:
        self._reconfigure_port()