from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_position_base_(self, enumerated, vertical):
    location = self.cur_token_location_
    self.expect_keyword_('base')
    if enumerated:
        raise FeatureLibError('"enumerate" is not allowed with mark-to-base attachment positioning', location)
    base = self.parse_glyphclass_(accept_glyphname=True)
    marks = self.parse_anchor_marks_()
    self.expect_symbol_(';')
    return self.ast.MarkBasePosStatement(base, marks, location=location)