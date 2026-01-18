from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def encodeFloat(f):
    if f == 0.0:
        return realZeroBytes
    s = '%.8G' % f
    if s[:2] == '0.':
        s = s[1:]
    elif s[:3] == '-0.':
        s = '-' + s[2:]
    nibbles = []
    while s:
        c = s[0]
        s = s[1:]
        if c == 'E':
            c2 = s[:1]
            if c2 == '-':
                s = s[1:]
                c = 'E-'
            elif c2 == '+':
                s = s[1:]
        nibbles.append(realNibblesDict[c])
    nibbles.append(15)
    if len(nibbles) % 2:
        nibbles.append(15)
    d = bytechr(30)
    for i in range(0, len(nibbles), 2):
        d = d + bytechr(nibbles[i] << 4 | nibbles[i + 1])
    return d