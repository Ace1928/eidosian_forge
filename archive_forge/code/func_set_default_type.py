import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def set_default_type(self, ctype):
    """Set the `default' content type.

        ctype should be either "text/plain" or "message/rfc822", although this
        is not enforced.  The default content type is not stored in the
        Content-Type header.
        """
    self._default_type = ctype