import codecs
import datetime
import locale
from decimal import Decimal
from types import NoneType
from urllib.parse import quote
from django.utils.functional import Promise
def iri_to_uri(iri):
    """
    Convert an Internationalized Resource Identifier (IRI) portion to a URI
    portion that is suitable for inclusion in a URL.

    This is the algorithm from RFC 3987 Section 3.1, slightly simplified since
    the input is assumed to be a string rather than an arbitrary byte stream.

    Take an IRI (string or UTF-8 bytes, e.g. '/I ♥ Django/' or
    b'/I â\x99¥ Django/') and return a string containing the encoded
    result with ASCII chars only (e.g. '/I%20%E2%99%A5%20Django/').
    """
    if iri is None:
        return iri
    elif isinstance(iri, Promise):
        iri = str(iri)
    return quote(iri, safe="/#%[]=:;$&()+,!?*@'~")