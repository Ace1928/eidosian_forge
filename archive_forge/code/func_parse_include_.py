from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_include_(self):
    assert self.cur_token_ == 'include'
    location = self.cur_token_location_
    filename = self.expect_filename_()
    return ast.IncludeStatement(filename, location=location)