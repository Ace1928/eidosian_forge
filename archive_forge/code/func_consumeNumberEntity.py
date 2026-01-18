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
def consumeNumberEntity(self, isHex):
    """This function returns either U+FFFD or the character based on the
        decimal or hexadecimal representation. It also discards ";" if present.
        If not present self.tokenQueue.append({"type": tokenTypes["ParseError"]}) is invoked.
        """
    allowed = digits
    radix = 10
    if isHex:
        allowed = hexDigits
        radix = 16
    charStack = []
    c = self.stream.char()
    while c in allowed and c is not EOF:
        charStack.append(c)
        c = self.stream.char()
    charAsInt = int(''.join(charStack), radix)
    if charAsInt in replacementCharacters:
        char = replacementCharacters[charAsInt]
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'illegal-codepoint-for-numeric-entity', 'datavars': {'charAsInt': charAsInt}})
    elif 55296 <= charAsInt <= 57343 or charAsInt > 1114111:
        char = '�'
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'illegal-codepoint-for-numeric-entity', 'datavars': {'charAsInt': charAsInt}})
    else:
        if 1 <= charAsInt <= 8 or 14 <= charAsInt <= 31 or 127 <= charAsInt <= 159 or (64976 <= charAsInt <= 65007) or (charAsInt in frozenset([11, 65534, 65535, 131070, 131071, 196606, 196607, 262142, 262143, 327678, 327679, 393214, 393215, 458750, 458751, 524286, 524287, 589822, 589823, 655358, 655359, 720894, 720895, 786430, 786431, 851966, 851967, 917502, 917503, 983038, 983039, 1048574, 1048575, 1114110, 1114111])):
            self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'illegal-codepoint-for-numeric-entity', 'datavars': {'charAsInt': charAsInt}})
        try:
            char = chr(charAsInt)
        except ValueError:
            v = charAsInt - 65536
            char = chr(55296 | v >> 10) + chr(56320 | v & 1023)
    if c != ';':
        self.tokenQueue.append({'type': tokenTypes['ParseError'], 'data': 'numeric-entity-without-semicolon'})
        self.stream.unget(c)
    return char