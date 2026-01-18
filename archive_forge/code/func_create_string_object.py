import codecs
from typing import Dict, List, Tuple, Union
from .._codecs import _pdfdoc_encoding
from .._utils import StreamType, b_, logger_warning, read_non_whitespace
from ..errors import STREAM_TRUNCATED_PREMATURELY, PdfStreamError
from ._base import ByteStringObject, TextStringObject
def create_string_object(string: Union[str, bytes], forced_encoding: Union[None, str, List[str], Dict[int, str]]=None) -> Union[TextStringObject, ByteStringObject]:
    """
    Create a ByteStringObject or a TextStringObject from a string to represent the string.

    Args:
        string: The data being used
        forced_encoding: Typically None, or an encoding string

    Returns:
        A ByteStringObject

    Raises:
        TypeError: If string is not of type str or bytes.
    """
    if isinstance(string, str):
        return TextStringObject(string)
    elif isinstance(string, bytes):
        if isinstance(forced_encoding, (list, dict)):
            out = ''
            for x in string:
                try:
                    out += forced_encoding[x]
                except Exception:
                    out += bytes((x,)).decode('charmap')
            return TextStringObject(out)
        elif isinstance(forced_encoding, str):
            if forced_encoding == 'bytes':
                return ByteStringObject(string)
            return TextStringObject(string.decode(forced_encoding))
        else:
            try:
                if string.startswith((codecs.BOM_UTF16_BE, codecs.BOM_UTF16_LE)):
                    retval = TextStringObject(string.decode('utf-16'))
                    retval.autodetect_utf16 = True
                    return retval
                else:
                    retval = TextStringObject(decode_pdfdocencoding(string))
                    retval.autodetect_pdfdocencoding = True
                    return retval
            except UnicodeDecodeError:
                return ByteStringObject(string)
    else:
        raise TypeError('create_string_object should have str or unicode arg')