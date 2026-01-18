from io import BytesIO
import struct
import time
import dns.exception
import dns.name
import dns.node
import dns.rdataset
import dns.rdata
import dns.rdatatype
import dns.rdataclass
from ._compat import string_types
def algorithm_to_text(value):
    """Convert a DNSSEC algorithm value to text

    Returns a ``str``.
    """
    text = _algorithm_by_value.get(value)
    if text is None:
        text = str(value)
    return text