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
def attributeNameState(self):
    data = self.stream.char()
    leavingThisState = True
    emitToken = False
    if data == '=':
        self.state = self.beforeAttributeValueState
    elif data in asciiLetters:
        self.currentToken['data'][-1][0] += data + self.stream.charsUntil(asciiLetters, True)
        leavingThisState = False
    elif data == '>':
        emitToken = True
    elif data in spaceCharacters:
        self.state = self.afterAttributeNameState
    elif data == '/':
        self.state = self.selfClosingStartTagState
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.currentToken['data'][-1][0] += '�'
        leavingThisState = False
    elif data in ("'", '"', '<'):
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-character-in-attribute-name'})
        self.currentToken['data'][-1][0] += data
        leavingThisState = False
    elif data is EOF:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'eof-in-attribute-name'})
        self.state = self.dataState
    else:
        self.currentToken['data'][-1][0] += data
        leavingThisState = False
    if leavingThisState:
        self.currentToken['data'][-1][0] = self.currentToken['data'][-1][0].translate(asciiUpper2Lower)
        for name, _ in self.currentToken['data'][:-1]:
            if self.currentToken['data'][-1][0] == name:
                self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'duplicate-attribute'})
                break
        if emitToken:
            self.emitCurrentToken()
    return True