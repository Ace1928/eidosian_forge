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
def scriptDataEscapedDashState(self):
    data = self.stream.char()
    if data == '-':
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '-'})
        self.state = self.scriptDataEscapedDashDashState
    elif data == '<':
        self.state = self.scriptDataEscapedLessThanSignState
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '�'})
        self.state = self.scriptDataEscapedState
    elif data == EOF:
        self.state = self.dataState
    else:
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': data})
        self.state = self.scriptDataEscapedState
    return True