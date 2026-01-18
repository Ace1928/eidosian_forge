from binascii import unhexlify
from math import ceil
from typing import Any, Dict, List, Tuple, Union, cast
from ._codecs import adobe_glyphs, charset_encoding
from ._utils import b_, logger_error, logger_warning
from .generic import (
def prepare_cm(ft: DictionaryObject) -> bytes:
    tu = ft['/ToUnicode']
    cm: bytes
    if isinstance(tu, StreamObject):
        cm = b_(cast(DecodedStreamObject, ft['/ToUnicode']).get_data())
    elif isinstance(tu, str) and tu.startswith('/Identity'):
        cm = b'beginbfrange\n<0000> <0001> <0000>\nendbfrange'
    if isinstance(cm, str):
        cm = cm.encode()
    cm = cm.strip().replace(b'beginbfchar', b'\nbeginbfchar\n').replace(b'endbfchar', b'\nendbfchar\n').replace(b'beginbfrange', b'\nbeginbfrange\n').replace(b'endbfrange', b'\nendbfrange\n').replace(b'<<', b'\n{\n').replace(b'>>', b'\n}\n')
    ll = cm.split(b'<')
    for i in range(len(ll)):
        j = ll[i].find(b'>')
        if j >= 0:
            if j == 0:
                content = b'.'
            else:
                content = ll[i][:j].replace(b' ', b'')
            ll[i] = content + b' ' + ll[i][j + 1:]
    cm = b' '.join(ll).replace(b'[', b' [ ').replace(b']', b' ]\n ').replace(b'\r', b'\n')
    return cm