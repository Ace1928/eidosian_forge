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
def tagNameState(self):
    data = self.stream.char()
    if data in spaceCharacters:
        self.state = self.beforeAttributeNameState
    elif data == '>':
        self.emitCurrentToken()
    elif data is EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'eof-in-tag-name'})
        self.state = self.dataState
    elif data == '/':
        self.state = self.selfClosingStartTagState
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.currentToken['name'] += '�'
    else:
        self.currentToken['name'] += data
    return True