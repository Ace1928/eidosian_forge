import base64
import binascii
import collections
import html
from django.conf import settings
from django.core.exceptions import (
from django.core.files.uploadhandler import SkipFile, StopFutureHandlers, StopUpload
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.http import parse_header_parameters
from django.utils.regex_helper import _lazy_re_compile
def parse_boundary_stream(stream, max_header_size):
    """
    Parse one and exactly one stream that encapsulates a boundary.
    """
    chunk = stream.read(max_header_size)
    header_end = chunk.find(b'\r\n\r\n')
    if header_end == -1:
        stream.unget(chunk)
        return (RAW, {}, stream)
    header = chunk[:header_end]
    stream.unget(chunk[header_end + 4:])
    TYPE = RAW
    outdict = {}
    for line in header.split(b'\r\n'):
        try:
            main_value_pair, params = parse_header_parameters(line.decode())
            name, value = main_value_pair.split(':', 1)
            params = {k: v.encode() for k, v in params.items()}
        except ValueError:
            continue
        if name == 'content-disposition':
            TYPE = FIELD
            if params.get('filename'):
                TYPE = FILE
        outdict[name] = (value, params)
    if TYPE == RAW:
        stream.unget(chunk)
    return (TYPE, outdict, stream)