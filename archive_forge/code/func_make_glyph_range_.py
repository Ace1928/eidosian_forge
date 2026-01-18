from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def make_glyph_range_(self, location, start, limit):
    """(location, "a.sc", "d.sc") --> ["a.sc", "b.sc", "c.sc", "d.sc"]"""
    result = list()
    if len(start) != len(limit):
        raise FeatureLibError('Bad range: "%s" and "%s" should have the same length' % (start, limit), location)
    rev = self.reverse_string_
    prefix = os.path.commonprefix([start, limit])
    suffix = rev(os.path.commonprefix([rev(start), rev(limit)]))
    if len(suffix) > 0:
        start_range = start[len(prefix):-len(suffix)]
        limit_range = limit[len(prefix):-len(suffix)]
    else:
        start_range = start[len(prefix):]
        limit_range = limit[len(prefix):]
    if start_range >= limit_range:
        raise FeatureLibError('Start of range must be smaller than its end', location)
    uppercase = re.compile('^[A-Z]$')
    if uppercase.match(start_range) and uppercase.match(limit_range):
        for c in range(ord(start_range), ord(limit_range) + 1):
            result.append('%s%c%s' % (prefix, c, suffix))
        return result
    lowercase = re.compile('^[a-z]$')
    if lowercase.match(start_range) and lowercase.match(limit_range):
        for c in range(ord(start_range), ord(limit_range) + 1):
            result.append('%s%c%s' % (prefix, c, suffix))
        return result
    digits = re.compile('^[0-9]{1,3}$')
    if digits.match(start_range) and digits.match(limit_range):
        for i in range(int(start_range, 10), int(limit_range, 10) + 1):
            number = ('000' + str(i))[-len(start_range):]
            result.append('%s%s%s' % (prefix, number, suffix))
        return result
    raise FeatureLibError('Bad range: "%s-%s"' % (start, limit), location)