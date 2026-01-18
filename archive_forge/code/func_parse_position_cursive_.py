from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_position_cursive_(self, enumerated, vertical):
    location = self.cur_token_location_
    self.expect_keyword_('cursive')
    if enumerated:
        raise FeatureLibError('"enumerate" is not allowed with cursive attachment positioning', location)
    glyphclass = self.parse_glyphclass_(accept_glyphname=True)
    entryAnchor = self.parse_anchor_()
    exitAnchor = self.parse_anchor_()
    self.expect_symbol_(';')
    return self.ast.CursivePosStatement(glyphclass, entryAnchor, exitAnchor, location=location)