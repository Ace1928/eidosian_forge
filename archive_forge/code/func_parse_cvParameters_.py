from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_cvParameters_(self, tag):
    assert self.cur_token_ == 'cvParameters', self.cur_token_
    block = self.ast.NestedBlock(tag, self.cur_token_, location=self.cur_token_location_)
    self.expect_symbol_('{')
    for symtab in self.symbol_tables_:
        symtab.enter_scope()
    statements = block.statements
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_(comments=True)
        if self.cur_token_type_ is Lexer.COMMENT:
            statements.append(self.ast.Comment(self.cur_token_, location=self.cur_token_location_))
        elif self.is_cur_keyword_({'FeatUILabelNameID', 'FeatUITooltipTextNameID', 'SampleTextNameID', 'ParamUILabelNameID'}):
            statements.append(self.parse_cvNameIDs_(tag, self.cur_token_))
        elif self.is_cur_keyword_('Character'):
            statements.append(self.parse_cvCharacter_(tag))
        elif self.cur_token_ == ';':
            continue
        else:
            raise FeatureLibError('Expected statement: got {} {}'.format(self.cur_token_type_, self.cur_token_), self.cur_token_location_)
    self.expect_symbol_('}')
    for symtab in self.symbol_tables_:
        symtab.exit_scope()
    self.expect_symbol_(';')
    return block