from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_variable_scalar_(self):
    self.advance_lexer_()
    scalar = VariableScalar()
    while True:
        if self.cur_token_type_ == Lexer.SYMBOL and self.cur_token_ == ')':
            break
        location, value = self.expect_master_()
        scalar.add_value(location, value)
    return scalar