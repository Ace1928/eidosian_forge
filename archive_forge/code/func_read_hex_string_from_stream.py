import codecs
from typing import Dict, List, Tuple, Union
from .._codecs import _pdfdoc_encoding
from .._utils import StreamType, b_, logger_warning, read_non_whitespace
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfStreamError
from ._base import ByteStringObject, TextStringObject
def read_hex_string_from_stream(stream: StreamType, forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> Union['TextStringObject', 'ByteStringObject']:
    stream.read(1)
    txt = ''
    x = b''
    while True:
        tok = read_non_whitespace(stream)
        if not tok:
            raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
        if tok == b'>':
            break
        x += tok
        if len(x) == 2:
            txt += chr(int(x, base=16))
            x = b''
    if len(x) == 1:
        x += b'0'
    if len(x) == 2:
        txt += chr(int(x, base=16))
    return create_string_object(b_(txt), forced_encoding)