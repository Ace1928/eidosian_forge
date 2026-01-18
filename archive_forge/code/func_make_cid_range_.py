from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def make_cid_range_(self, location, start, limit):
    """(location, 999, 1001) --> ["cid00999", "cid01000", "cid01001"]"""
    result = list()
    if start > limit:
        raise FeatureLibError('Bad range: start should be less than limit', location)
    for cid in range(start, limit + 1):
        result.append('cid%05d' % cid)
    return result