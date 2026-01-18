from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_keyword_(self, keyword):
    self.advance_lexer_()
    if self.cur_token_type_ is Lexer.NAME and self.cur_token_ == keyword:
        return self.cur_token_
    raise FeatureLibError('Expected "%s"' % keyword, self.cur_token_location_)