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
def tagOpenState(self):
    data = self.stream.char()
    if data == '!':
        self.state = self.markupDeclarationOpenState
    elif data == '/':
        self.state = self.closeTagOpenState
    elif data in asciiLetters:
        self.currentToken = {'type': tokenTypes['StartTag'], 'name': data, 'data': [], 'selfClosing': False, 'selfClosingAcknowledged': False}
        self.state = self.tagNameState
    elif data == '>':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-tag-name-but-got-right-bracket'})
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '<>'})
        self.state = self.dataState
    elif data == '?':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-tag-name-but-got-question-mark'})
        self.stream.unget(data)
        self.state = self.bogusCommentState
    else:
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-tag-name'})
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '<'})
        self.stream.unget(data)
        self.state = self.dataState
    return True