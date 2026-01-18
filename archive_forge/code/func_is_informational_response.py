import collections
import re
from string import whitespace
import sys
from hpack import HeaderTuple, NeverIndexedHeaderTuple
from .exceptions import ProtocolError, FlowControlError
def is_informational_response(headers):
    """
    Searches a header block for a :status header to confirm that a given
    collection of headers are an informational response. Assumes the header
    block is well formed: that is, that the HTTP/2 special headers are first
    in the block, and so that it can stop looking when it finds the first
    header field whose name does not begin with a colon.

    :param headers: The HTTP/2 header block.
    :returns: A boolean indicating if this is an informational response.
    """
    for n, v in headers:
        if isinstance(n, bytes):
            sigil = b':'
            status = b':status'
            informational_start = b'1'
        else:
            sigil = u':'
            status = u':status'
            informational_start = u'1'
        if not n.startswith(sigil):
            return False
        if n != status:
            continue
        return v.startswith(informational_start)