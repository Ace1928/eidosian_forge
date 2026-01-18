import codecs
import datetime
import locale
from decimal import Decimal
from types import NoneType
from urllib.parse import quote
from django.utils.functional import Promise
def filepath_to_uri(path):
    """Convert a file system path to a URI portion that is suitable for
    inclusion in a URL.

    Encode certain chars that would normally be recognized as special chars
    for URIs. Do not encode the ' character, as it is a valid character
    within URIs. See the encodeURIComponent() JavaScript function for details.
    """
    if path is None:
        return path
    return quote(str(path).replace('\\', '/'), safe="/~!*()'")