import codecs
from typing import Dict, List, Tuple, Union
from .._codecs import _pdfdoc_encoding
from .._utils import StreamType, b_, logger_warning, read_non_whitespace
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfStreamError
from ._base import ByteStringObject, TextStringObject
def read_string_from_stream(stream: StreamType, forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> Union['TextStringObject', 'ByteStringObject']:
    tok = stream.read(1)
    parens = 1
    txt = []
    while True:
        tok = stream.read(1)
        if not tok:
            raise PdfStreamError(STREAM_TRUNCATED_PREMATURELY)
        if tok == b'(':
            parens += 1
        elif tok == b')':
            parens -= 1
            if parens == 0:
                break
        elif tok == b'\\':
            tok = stream.read(1)
            escape_dict = {b'n': b'\n', b'r': b'\r', b't': b'\t', b'b': b'\x08', b'f': b'\x0c', b'c': b'\\c', b'(': b'(', b')': b')', b'/': b'/', b'\\': b'\\', b' ': b' ', b'%': b'%', b'<': b'<', b'>': b'>', b'[': b'[', b']': b']', b'#': b'#', b'_': b'_', b'&': b'&', b'$': b'$'}
            try:
                tok = escape_dict[tok]
            except KeyError:
                if b'0' <= tok <= b'7':
                    for _ in range(2):
                        ntok = stream.read(1)
                        if b'0' <= ntok <= b'7':
                            tok += ntok
                        else:
                            stream.seek(-1, 1)
                            break
                    tok = b_(chr(int(tok, base=8)))
                elif tok in b'\n\r':
                    tok = stream.read(1)
                    if tok not in b'\n\r':
                        stream.seek(-1, 1)
                    tok = b''
                else:
                    msg = f'Unexpected escaped string: {tok.decode('utf-8', 'ignore')}'
                    logger_warning(msg, __name__)
        txt.append(tok)
    return create_string_object(b''.join(txt), forced_encoding)