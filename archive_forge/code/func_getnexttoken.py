from fontTools.misc.textTools import bytechr, byteord, bytesjoin, tobytes, tostr
from fontTools.misc import eexec
from .psOperators import (
import re
from collections.abc import Callable
from string import whitespace
import logging
def getnexttoken(self, len=len, ps_special=ps_special, stringmatch=stringRE.match, hexstringmatch=hexstringRE.match, commentmatch=commentRE.match, endmatch=endofthingRE.match):
    self.skipwhite()
    if self.pos >= self.len:
        return (None, None)
    pos = self.pos
    buf = self.buf
    char = bytechr(byteord(buf[pos]))
    if char in ps_special:
        if char in b'{}[]':
            tokentype = 'do_special'
            token = char
        elif char == b'%':
            tokentype = 'do_comment'
            _, nextpos = commentmatch(buf, pos).span()
            token = buf[pos:nextpos]
        elif char == b'(':
            tokentype = 'do_string'
            m = stringmatch(buf, pos)
            if m is None:
                raise PSTokenError('bad string at character %d' % pos)
            _, nextpos = m.span()
            token = buf[pos:nextpos]
        elif char == b'<':
            tokentype = 'do_hexstring'
            m = hexstringmatch(buf, pos)
            if m is None:
                raise PSTokenError('bad hexstring at character %d' % pos)
            _, nextpos = m.span()
            token = buf[pos:nextpos]
        else:
            raise PSTokenError('bad token at character %d' % pos)
    else:
        if char == b'/':
            tokentype = 'do_literal'
            m = endmatch(buf, pos + 1)
        else:
            tokentype = ''
            m = endmatch(buf, pos)
        if m is None:
            raise PSTokenError('bad token at character %d' % pos)
        _, nextpos = m.span()
        token = buf[pos:nextpos]
    self.pos = pos + len(token)
    token = tostr(token, encoding=self.encoding)
    return (tokentype, token)