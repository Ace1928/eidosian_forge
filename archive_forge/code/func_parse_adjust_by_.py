import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_adjust_by_(self):
    self.advance_lexer_()
    assert self.is_cur_keyword_('ADJUST_BY')
    adjustment = self.expect_number_()
    self.expect_keyword_('AT')
    size = self.expect_number_()
    return (adjustment, size)