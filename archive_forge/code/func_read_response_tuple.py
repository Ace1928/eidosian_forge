from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def read_response_tuple(self, expect_body=False):
    """Read a response tuple from the wire."""
    self._wait_for_response_args()
    if not expect_body:
        self._wait_for_response_end()
    if 'hpss' in debug.debug_flags:
        mutter('   result:   %r', self.args)
    if self.status == b'E':
        self._wait_for_response_end()
        _raise_smart_server_error(self.args)
    return tuple(self.args)