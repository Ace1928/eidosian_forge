import base64
import calendar
import datetime
import sys
import six
from six.moves import urllib
from google.auth import exceptions
def padded_urlsafe_b64decode(value):
    """Decodes base64 strings lacking padding characters.

    Google infrastructure tends to omit the base64 padding characters.

    Args:
        value (Union[str, bytes]): The encoded value.

    Returns:
        bytes: The decoded value
    """
    b64string = to_bytes(value)
    padded = b64string + b'=' * (-len(b64string) % 4)
    return base64.urlsafe_b64decode(padded)