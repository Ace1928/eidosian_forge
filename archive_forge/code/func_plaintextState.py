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
def plaintextState(self):
    data = self.stream.char()
    if data == EOF:
        return False
    elif data == '\x00':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'invalid-codepoint'})
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': '�'})
    else:
        self.tokenQueue.append({'type': tokenTypes['Characters'], 'data': data + self.stream.charsUntil('\x00')})
    return True