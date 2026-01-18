from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_subtable_(self):
    assert self.is_cur_keyword_('subtable')
    location = self.cur_token_location_
    self.expect_symbol_(';')
    return self.ast.SubtableStatement(location=location)