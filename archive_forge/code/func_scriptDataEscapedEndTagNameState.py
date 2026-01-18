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
def scriptDataEscapedEndTagNameState(self):
    appropriate = self.currentToken and self.currentToken['name'].lower() == self.temporaryBuffer.lower()
    data = self.stream.char()
    if data in spaceCharacters and appropriate:
        self.currentToken = {'type': tokenTypes['EndTag'], 'name': self.temporaryBuffer, 'data': [], 'selfClosing': False}
        self.state = self.beforeAttributeNameState
    elif data == '/' and appropriate:
        self.currentToken = {'type': tokenTypes['EndTag'], 'name': self.temporaryBuffer, 'data': [], 'selfClosing': False}
        self.state = self.selfClosingStartTagState
    elif data == '>' and appropriate:
        self.currentToken = {'type': tokenTypes['EndTag'], 'name': self.temporaryBuffer, 'data': [], 'selfClosing': False}
        self.emitCurrentToken()
        self.state = self.dataState
    elif data in asciiLetters:
        self.temporaryBuffer += data
    else:
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '</' + self.temporaryBuffer})
        self.stream.unget(data)
        self.state = self.scriptDataEscapedState
    return True