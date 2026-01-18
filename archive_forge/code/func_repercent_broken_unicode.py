import codecs
import datetime
import locale
from decimal import Decimal
from types import NoneType
from urllib.parse import quote
from django.utils.functional import Promise
def repercent_broken_unicode(path):
    """
    As per RFC 3987 Section 3.2, step three of converting a URI into an IRI,
    repercent-encode any octet produced that is not part of a strictly legal
    UTF-8 octet sequence.
    """
    changed_parts = []
    while True:
        try:
            path.decode()
        except UnicodeDecodeError as e:
            repercent = quote(path[e.start:e.end], safe=b"/#%[]=:;$&()+,!?*@'~")
            changed_parts.append(path[:e.start] + repercent.encode())
            path = path[e.end:]
        else:
            return b''.join(changed_parts) + path