from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_STAT_location(self):
    values = []
    tag = self.expect_tag_()
    if len(tag.strip()) != 4:
        raise FeatureLibError(f'Axis tag {self.cur_token_} must be 4 characters', self.cur_token_location_)
    while self.next_token_ != ';':
        if self.next_token_type_ is Lexer.FLOAT:
            value = self.expect_float_()
            values.append(value)
        elif self.next_token_type_ is Lexer.NUMBER:
            value = self.expect_number_()
            values.append(value)
        else:
            raise FeatureLibError(f'Unexpected value "{self.next_token_}". Expected integer or float.', self.next_token_location_)
    if len(values) == 3:
        nominal, min_val, max_val = values
        if nominal < min_val or nominal > max_val:
            raise FeatureLibError(f'Default value {nominal} is outside of specified range {min_val}-{max_val}.', self.next_token_location_)
    return self.ast.AxisValueLocationStatement(tag, values)