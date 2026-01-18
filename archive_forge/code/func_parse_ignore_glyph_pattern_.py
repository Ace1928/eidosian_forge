from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_ignore_glyph_pattern_(self, sub):
    location = self.cur_token_location_
    prefix, glyphs, lookups, values, suffix, hasMarks = self.parse_glyph_pattern_(vertical=False)
    if any(lookups):
        raise FeatureLibError(f'No lookups can be specified for "ignore {sub}"', location)
    if not hasMarks:
        error = FeatureLibError(f'Ambiguous "ignore {sub}", there should be least one marked glyph', location)
        log.warning(str(error))
        suffix, glyphs = (glyphs[1:], glyphs[0:1])
    chainContext = (prefix, glyphs, suffix)
    return chainContext