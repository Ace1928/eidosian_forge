from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_FontRevision_(self):
    assert self.cur_token_ == 'FontRevision', self.cur_token_
    location, version = (self.cur_token_location_, self.expect_float_())
    self.expect_symbol_(';')
    if version <= 0:
        raise FeatureLibError('Font revision numbers must be positive', location)
    return self.ast.FontRevisionStatement(version, location=location)