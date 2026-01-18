from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_table_name_(self, table):
    statements = table.statements
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_(comments=True)
        if self.cur_token_type_ is Lexer.COMMENT:
            statements.append(self.ast.Comment(self.cur_token_, location=self.cur_token_location_))
        elif self.is_cur_keyword_('nameid'):
            statement = self.parse_nameid_()
            if statement:
                statements.append(statement)
        elif self.cur_token_ == ';':
            continue
        else:
            raise FeatureLibError('Expected nameid', self.cur_token_location_)