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
def consumeEntity(self, allowedChar=None, fromAttribute=False):
    output = '&'
    charStack = [self.stream.char()]
    if charStack[0] in spaceCharacters or charStack[0] in (EOF, '<', '&') or (allowedChar is not None and allowedChar == charStack[0]):
        self.stream.unget(charStack[0])
    elif charStack[0] == '#':
        hex = False
        charStack.append(self.stream.char())
        if charStack[-1] in ('x', 'X'):
            hex = True
            charStack.append(self.stream.char())
        if hex and charStack[-1] in hexDigits or (not hex and charStack[-1] in digits):
            self.stream.unget(charStack[-1])
            output = self.consumeNumberEntity(hex)
        else:
            self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-numeric-entity'})
            self.stream.unget(charStack.pop())
            output = '&' + ''.join(charStack)
    else:
        while charStack[-1] is not EOF:
            if not entitiesTrie.has_keys_with_prefix(''.join(charStack)):
                break
            charStack.append(self.stream.char())
        try:
            entityName = entitiesTrie.longest_prefix(''.join(charStack[:-1]))
            entityLength = len(entityName)
        except KeyError:
            entityName = None
        if entityName is not None:
            if entityName[-1] != ';':
                self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'named-entity-without-semicolon'})
            if entityName[-1] != ';' and fromAttribute and (charStack[entityLength] in asciiLetters or charStack[entityLength] in digits or charStack[entityLength] == '='):
                self.stream.unget(charStack.pop())
                output = '&' + ''.join(charStack)
            else:
                output = entities[entityName]
                self.stream.unget(charStack.pop())
                output += ''.join(charStack[entityLength:])
        else:
            self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'expected-named-entity'})
            self.stream.unget(charStack.pop())
            output = '&' + ''.join(charStack)
    if fromAttribute:
        self.currentToken['data'][-1][1] += output
    else:
        if output in spaceCharacters:
            tokenType = 'SpaceCharacters'
        else:
            tokenType = 'Characters'
        self.tokenQueue.append({'type': tokenTypes[tokenType], 'data': output})