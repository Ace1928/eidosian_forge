from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_symbol_(self, symbol):
    self.advance_lexer_()
    if self.cur_token_type_ is Lexer.SYMBOL and self.cur_token_ == symbol:
        return symbol
    raise FeatureLibError("Expected '%s'" % symbol, self.cur_token_location_)