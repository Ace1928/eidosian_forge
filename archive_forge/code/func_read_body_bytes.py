from collections import deque
from io import BytesIO
from ... import debug, errors
from ...trace import mutter
def read_body_bytes(self, count=-1):
    """Read bytes from the body, decoding into a byte stream.

        We read all bytes at once to ensure we've checked the trailer for
        errors, and then feed the buffer back as read_body_bytes is called.

        Like the builtin file.read in Python, a count of -1 (the default) means
        read the entire body.
        """
    if self._body is None:
        self._wait_for_response_end()
        body_bytes = b''.join(self._bytes_parts)
        if 'hpss' in debug.debug_flags:
            mutter('              %d body bytes read', len(body_bytes))
        self._body = BytesIO(body_bytes)
        self._bytes_parts = None
    return self._body.read(count)