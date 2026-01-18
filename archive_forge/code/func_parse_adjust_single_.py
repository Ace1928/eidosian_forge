import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_adjust_single_(self):
    assert self.is_cur_keyword_('ADJUST_SINGLE')
    location = self.cur_token_location_
    adjust_single = []
    while self.next_token_ != 'END_ADJUST':
        coverages = self.parse_coverage_()
        self.expect_keyword_('BY')
        pos = self.parse_pos_()
        adjust_single.append((coverages, pos))
    self.expect_keyword_('END_ADJUST')
    position = ast.PositionAdjustSingleDefinition(adjust_single, location=location)
    return position