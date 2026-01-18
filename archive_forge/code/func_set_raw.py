import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def set_raw(self, name, value):
    """Store name and value in the model without modification.

        This is an "internal" API, intended only for use by a parser.
        """
    self._headers.append((name, value))