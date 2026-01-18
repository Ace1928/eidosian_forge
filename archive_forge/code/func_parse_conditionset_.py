from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_conditionset_(self):
    name = self.expect_name_()
    conditions = {}
    self.expect_symbol_('{')
    while self.next_token_ != '}':
        self.advance_lexer_()
        if self.cur_token_type_ is not Lexer.NAME:
            raise FeatureLibError('Expected an axis name', self.cur_token_location_)
        axis = self.cur_token_
        if axis in conditions:
            raise FeatureLibError(f'Repeated condition for axis {axis}', self.cur_token_location_)
        if self.next_token_type_ is Lexer.FLOAT:
            min_value = self.expect_float_()
        elif self.next_token_type_ is Lexer.NUMBER:
            min_value = self.expect_number_(variable=False)
        if self.next_token_type_ is Lexer.FLOAT:
            max_value = self.expect_float_()
        elif self.next_token_type_ is Lexer.NUMBER:
            max_value = self.expect_number_(variable=False)
        self.expect_symbol_(';')
        conditions[axis] = (min_value, max_value)
    self.expect_symbol_('}')
    finalname = self.expect_name_()
    if finalname != name:
        raise FeatureLibError('Expected "%s"' % name, self.cur_token_location_)
    return self.ast.ConditionsetStatement(name, conditions)