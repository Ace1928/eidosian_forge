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
def emitCurrentToken(self):
    """This method is a generic handler for emitting the tags. It also sets
        the state to "data" because that's what's needed after a token has been
        emitted.
        """
    token = self.currentToken
    if token['type'] in tagTokenTypes:
        token['name'] = token['name'].translate(asciiUpper2Lower)
        if token['type'] == tokenTypes['StartTag']:
            raw = token['data']
            data = attributeMap(raw)
            if len(raw) > len(data):
                data.update(raw[::-1])
            token['data'] = data
        if token['type'] == tokenTypes['EndTag']:
            if token['data']:
                self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'attributes-in-end-tag'})
            if token['selfClosing']:
                self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'self-closing-flag-on-end-tag'})
    self.tokenQueue.append(token)
    self.state = self.dataState