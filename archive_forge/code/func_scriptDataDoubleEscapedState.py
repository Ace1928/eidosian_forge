from __future__ import absolute_import, division, unicode_literals
from six import unichr as chr
from collections import deque, OrderedDict
from sys import version_info
from .constants import spaceCharacters
from .constants import entities
from .constants import asciiLetters, asciiUpper2Lower
from .constants import digits, hexDigits, EOF
from .constants import tokenTypes, tagTokenTypes
from .constants import replacementCharacters
from ._inputstream import HTMLInputStream
from ._trie import Trie
def scriptDataDoubleEscapedState(self):
    data = self.stream.char()
    if data == '-':
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '-'})
        self.state = self.scriptDataDoubleEscapedDashState
    elif data == '<':
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '<'})
        self.state = self.scriptDataDoubleEscapedLessThanSignState
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '�'})
    elif data == EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'eof-in-script-in-script'})
        self.state = self.dataState
    else:
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': data})
    return True