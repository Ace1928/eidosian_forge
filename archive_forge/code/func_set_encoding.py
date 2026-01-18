from __future__ import annotations
import codecs
import contextlib
import sys
import typing
import warnings
from contextlib import suppress
from urwid import str_util
def set_encoding(encoding: str) -> None:
    """
    Set the byte encoding to assume when processing strings and the
    encoding to use when converting unicode strings.
    """
    encoding = encoding.lower()
    global _target_encoding, _use_dec_special
    if encoding in {'utf-8', 'utf8', 'utf'}:
        str_util.set_byte_encoding('utf8')
        _use_dec_special = False
    elif encoding in {'euc-jp', 'euc-kr', 'euc-cn', 'euc-tw', 'gb2312', 'gbk', 'big5', 'cn-gb', 'uhc', 'eucjp', 'euckr', 'euccn', 'euctw', 'cncb'}:
        str_util.set_byte_encoding('wide')
        _use_dec_special = True
    else:
        str_util.set_byte_encoding('narrow')
        _use_dec_special = True
    _target_encoding = 'ascii'
    with contextlib.suppress(LookupError):
        if encoding:
            ''.encode(encoding)
            _target_encoding = encoding