from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_size_parameters_(self):
    assert self.is_cur_keyword_('parameters')
    location = self.cur_token_location_
    DesignSize = self.expect_decipoint_()
    SubfamilyID = self.expect_number_()
    RangeStart = 0.0
    RangeEnd = 0.0
    if self.next_token_type_ in (Lexer.NUMBER, Lexer.FLOAT) or SubfamilyID != 0:
        RangeStart = self.expect_decipoint_()
        RangeEnd = self.expect_decipoint_()
    self.expect_symbol_(';')
    return self.ast.SizeParameters(DesignSize, SubfamilyID, RangeStart, RangeEnd, location=location)