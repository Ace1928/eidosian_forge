import base64
import hashlib
import hmac
import re
import urllib.parse
from keystoneclient.i18n import _
def date_param():
    """Get the X-Amz-Date' value.

            The value can be either a header or parameter.

            Note AWS supports parsing the Date header also, but this is not
            currently supported here as it will require some format mangling
            So the X-Amz-Date value must be YYYYMMDDTHHMMSSZ format, then it
            can be used to match against the YYYYMMDD format provided in the
            credential scope.
            see:
            http://docs.aws.amazon.com/general/latest/gr/
            sigv4-date-handling.html
            """
    try:
        return headers['X-Amz-Date']
    except KeyError:
        return params.get('X-Amz-Date')