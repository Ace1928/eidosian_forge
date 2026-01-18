from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_glyph_(self):
    self.advance_lexer_()
    if self.cur_token_type_ is Lexer.NAME:
        return self.cur_token_.lstrip('\\')
    elif self.cur_token_type_ is Lexer.CID:
        return 'cid%05d' % self.cur_token_
    raise FeatureLibError('Expected a glyph name or CID', self.cur_token_location_)