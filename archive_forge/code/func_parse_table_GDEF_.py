from fontTools.feaLib.error import FeatureLibError
from fontTools.feaLib.lexer import Lexer, IncludingLexer, NonIncludingLexer
from fontTools.feaLib.variableScalar import VariableScalar
from fontTools.misc.encodingTools import getEncoding
from fontTools.misc.textTools import bytechr, tobytes, tostr
import fontTools.feaLib.ast as ast
import logging
import os
import re
def parse_table_GDEF_(self, table):
    statements = table.statements
    while self.next_token_ != '}' or self.cur_comments_:
        self.advance_lexer_(comments=True)
        if self.cur_token_type_ is Lexer.COMMENT:
            statements.append(self.ast.Comment(self.cur_token_, location=self.cur_token_location_))
        elif self.is_cur_keyword_('Attach'):
            statements.append(self.parse_attach_())
        elif self.is_cur_keyword_('GlyphClassDef'):
            statements.append(self.parse_GlyphClassDef_())
        elif self.is_cur_keyword_('LigatureCaretByIndex'):
            statements.append(self.parse_ligatureCaretByIndex_())
        elif self.is_cur_keyword_('LigatureCaretByPos'):
            statements.append(self.parse_ligatureCaretByPos_())
        elif self.cur_token_ == ';':
            continue
        else:
            raise FeatureLibError('Expected Attach, LigatureCaretByIndex, or LigatureCaretByPos', self.cur_token_location_)