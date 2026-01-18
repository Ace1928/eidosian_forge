import os
from boto.vendored import six
from boto.vendored.six import BytesIO, StringIO
from boto.vendored.six.moves import filter, http_client, map, _thread, \
from boto.vendored.six.moves.queue import Queue
from boto.vendored.six.moves.urllib.parse import parse_qs, quote, unquote, \
from boto.vendored.six.moves.urllib.parse import unquote_plus
from boto.vendored.six.moves.urllib.request import urlopen
def parse_qs_safe(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace'):
    """Parse a query handling unicode arguments properly in Python 2."""
    is_text_type = isinstance(qs, six.text_type)
    if is_text_type:
        qs = qs.encode('ascii')
    qs_dict = parse_qs(qs, keep_blank_values, strict_parsing)
    if is_text_type:
        result = {}
        for name, value in qs_dict.items():
            decoded_name = name.decode(encoding, errors)
            decoded_value = [item.decode(encoding, errors) for item in value]
            result[decoded_name] = decoded_value
        return result
    return qs_dict