import binascii
import codecs
import hashlib
import re
from binascii import unhexlify
from math import log10
from typing import Any, Callable, ClassVar, Dict, Optional, Sequence, Union, cast
from .._codecs import _pdfdoc_encoding_rev
from .._protocols import PdfObjectProtocol, PdfWriterProtocol
from .._utils import (
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfReadError, PdfStreamError
@staticmethod
def read_from_stream(stream: StreamType, pdf: Any) -> 'NameObject':
    name = stream.read(1)
    if name != NameObject.surfix:
        raise PdfReadError('name read error')
    name += read_until_regex(stream, NameObject.delimiter_pattern)
    try:
        name = NameObject.unnumber(name)
        for enc in NameObject.CHARSETS:
            try:
                ret = name.decode(enc)
                return NameObject(ret)
            except Exception:
                pass
        raise UnicodeDecodeError('', name, 0, 0, 'Code Not Found')
    except (UnicodeEncodeError, UnicodeDecodeError) as e:
        if not pdf.strict:
            logger_warning(f'Illegal character in NameObject ({name!r}), you may need to adjust NameObject.CHARSETS', __name__)
            return NameObject(name.decode('charmap'))
        else:
            raise PdfReadError(f'Illegal character in NameObject ({name!r}). You may need to adjust NameObject.CHARSETS.') from e