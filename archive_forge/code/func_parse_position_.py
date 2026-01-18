from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_position_(self, enumerated, vertical):
    assert self.cur_token_ in {'position', 'pos'}
    if self.next_token_ == 'cursive':
        return self.parse_position_cursive_(enumerated, vertical)
    elif self.next_token_ == 'base':
        return self.parse_position_base_(enumerated, vertical)
    elif self.next_token_ == 'ligature':
        return self.parse_position_ligature_(enumerated, vertical)
    elif self.next_token_ == 'mark':
        return self.parse_position_mark_(enumerated, vertical)
    location = self.cur_token_location_
    prefix, glyphs, lookups, values, suffix, hasMarks = self.parse_glyph_pattern_(vertical)
    self.expect_symbol_(';')
    if any(lookups):
        if any(values):
            raise FeatureLibError('If "lookup" is present, no values must be specified', location)
        return self.ast.ChainContextPosStatement(prefix, glyphs, suffix, lookups, location=location)
    if not prefix and (not suffix) and (len(glyphs) == 2) and (not hasMarks):
        if values[0] is None:
            values.reverse()
        return self.ast.PairPosStatement(glyphs[0], values[0], glyphs[1], values[1], enumerated=enumerated, location=location)
    if enumerated:
        raise FeatureLibError('"enumerate" is only allowed with pair positionings', location)
    return self.ast.SinglePosStatement(list(zip(glyphs, values)), prefix, suffix, forceChain=hasMarks, location=location)