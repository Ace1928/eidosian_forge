import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def set_payload(self, payload, charset=None):
    """Set the payload to the given value.

        Optional charset sets the message's default character set.  See
        set_charset() for details.
        """
    if hasattr(payload, 'encode'):
        if charset is None:
            self._payload = payload
            return
        if not isinstance(charset, Charset):
            charset = Charset(charset)
        payload = payload.encode(charset.output_charset)
    if hasattr(payload, 'decode'):
        self._payload = payload.decode('ascii', 'surrogateescape')
    else:
        self._payload = payload
    if charset is not None:
        self.set_charset(charset)