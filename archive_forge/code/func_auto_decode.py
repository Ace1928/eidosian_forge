import codecs
import locale
import re
import sys
from typing import List, Tuple
def auto_decode(data: bytes) -> str:
    """Check a bytes string for a BOM to correctly detect the encoding

    Fallback to locale.getpreferredencoding(False) like open() on Python3"""
    for bom, encoding in BOMS:
        if data.startswith(bom):
            return data[len(bom):].decode(encoding)
    for line in data.split(b'\n')[:2]:
        if line[0:1] == b'#' and ENCODING_RE.search(line):
            result = ENCODING_RE.search(line)
            assert result is not None
            encoding = result.groups()[0].decode('ascii')
            return data.decode(encoding)
    return data.decode(locale.getpreferredencoding(False) or sys.getdefaultencoding())