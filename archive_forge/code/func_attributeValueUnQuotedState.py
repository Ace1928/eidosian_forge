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
def attributeValueUnQuotedState(self):
    data = self.stream.char()
    if data in spaceCharacters:
        self.state = self.beforeAttributeNameState
    elif data == '&':
        self.processEntityInAttribute('>')
    elif data == '>':
        self.emitCurrentToken()
    elif data in ('"', "'", '=', '<', '`'):
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'unexpected-character-in-unquoted-attribute-value'})
        self.currentToken['data'][-1][1] += data
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.currentToken['data'][-1][1] += '�'
    elif data is EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'eof-in-attribute-value-no-quotes'})
        self.state = self.dataState
    else:
        self.currentToken['data'][-1][1] += data + self.stream.charsUntil(frozenset(('&', '>', '"', "'", '=', '<', '`', '\x00')) | spaceCharacters)
    return True