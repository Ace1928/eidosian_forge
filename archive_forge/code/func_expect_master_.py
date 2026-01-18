from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def expect_master_(self):
    location = {}
    while True:
        if self.cur_token_type_ is not Lexer.NAME:
            raise FeatureLibError('Expected an axis name', self.cur_token_location_)
        axis = self.cur_token_
        self.advance_lexer_()
        if not (self.cur_token_type_ is Lexer.SYMBOL and self.cur_token_ == '='):
            raise FeatureLibError('Expected an equals sign', self.cur_token_location_)
        value = self.expect_number_()
        location[axis] = value
        if self.next_token_type_ is Lexer.NAME and self.next_token_[0] == ':':
            break
        self.advance_lexer_()
        if not (self.cur_token_type_ is Lexer.SYMBOL and self.cur_token_ == ','):
            raise FeatureLibError('Expected an comma or an equals sign', self.cur_token_location_)
        self.advance_lexer_()
    self.advance_lexer_()
    value = int(self.cur_token_[1:])
    self.advance_lexer_()
    return (location, value)