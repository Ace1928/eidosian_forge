from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def advance_lexer_(self, comments=False):
    if comments and self.cur_comments_:
        self.cur_token_type_ = Lexer.COMMENT
        self.cur_token_, self.cur_token_location_ = self.cur_comments_.pop(0)
        return
    else:
        self.cur_token_type_, self.cur_token_, self.cur_token_location_ = (self.next_token_type_, self.next_token_, self.next_token_location_)
    while True:
        try:
            self.next_token_type_, self.next_token_, self.next_token_location_ = next(self.lexer_)
        except StopIteration:
            self.next_token_type_, self.next_token_ = (None, None)
        if self.next_token_type_ != Lexer.COMMENT:
            break
        self.cur_comments_.append((self.next_token_, self.next_token_location_))