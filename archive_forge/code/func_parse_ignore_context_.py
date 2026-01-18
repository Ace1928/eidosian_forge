from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_ignore_context_(self, sub):
    location = self.cur_token_location_
    chainContext = [self.parse_ignore_glyph_pattern_(sub)]
    while self.next_token_ == ',':
        self.expect_symbol_(',')
        chainContext.append(self.parse_ignore_glyph_pattern_(sub))
    self.expect_symbol_(';')
    return chainContext