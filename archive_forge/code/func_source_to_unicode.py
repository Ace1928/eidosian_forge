import io
from io import TextIOWrapper, BytesIO
from pathlib import Path
import re
from tokenize import open, detect_encoding
def source_to_unicode(txt, errors='replace', skip_encoding_cookie=True):
    """Converts a bytes string with python source code to unicode.

    Unicode strings are passed through unchanged. Byte strings are checked
    for the python source file encoding cookie to determine encoding.
    txt can be either a bytes buffer or a string containing the source
    code.
    """
    if isinstance(txt, str):
        return txt
    if isinstance(txt, bytes):
        buffer = BytesIO(txt)
    else:
        buffer = txt
    try:
        encoding, _ = detect_encoding(buffer.readline)
    except SyntaxError:
        encoding = 'ascii'
    buffer.seek(0)
    with TextIOWrapper(buffer, encoding, errors=errors, line_buffering=True) as text:
        text.mode = 'r'
        if skip_encoding_cookie:
            return u''.join(strip_encoding_cookie(text))
        else:
            return text.read()