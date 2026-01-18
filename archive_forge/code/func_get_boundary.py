import binascii
import re
import quopri
from io import BytesIO, StringIO
from email import utils
from email import errors
from email._policybase import Policy, compat32
from email import charset as _charset
from email._encoded_words import decode_b
def get_boundary(self, failobj=None):
    """Return the boundary associated with the payload if present.

        The boundary is extracted from the Content-Type header's `boundary'
        parameter, and it is unquoted.
        """
    missing = object()
    boundary = self.get_param('boundary', missing)
    if boundary is missing:
        return failobj
    return utils.collapse_rfc2231_value(boundary).rstrip()