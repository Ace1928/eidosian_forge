import datetime
import sys
import encodings.idna
from urllib3.fields import RequestField
from urllib3.filepost import encode_multipart_formdata
from urllib3.util import parse_url
from urllib3.exceptions import (
from io import UnsupportedOperation
from .hooks import default_hooks
from .structures import CaseInsensitiveDict
from .auth import HTTPBasicAuth
from .cookies import cookiejar_from_dict, get_cookie_header, _copy_cookie_jar
from .exceptions import (
from .exceptions import JSONDecodeError as RequestsJSONDecodeError
from .exceptions import SSLError as RequestsSSLError
from ._internal_utils import to_native_string, unicode_is_ascii
from .utils import (
from .compat import (
from .compat import json as complexjson
from .status_codes import codes
@property
def is_permanent_redirect(self):
    """True if this Response one of the permanent versions of redirect."""
    return 'location' in self.headers and self.status_code in (codes.moved_permanently, codes.permanent_redirect)