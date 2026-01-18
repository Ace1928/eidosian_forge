import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_adjust_pair_(self):
    assert self.is_cur_keyword_('ADJUST_PAIR')
    location = self.cur_token_location_
    coverages_1 = []
    coverages_2 = []
    adjust_pair = {}
    while self.next_token_ == 'FIRST':
        self.advance_lexer_()
        coverage_1 = self.parse_coverage_()
        coverages_1.append(coverage_1)
    while self.next_token_ == 'SECOND':
        self.advance_lexer_()
        coverage_2 = self.parse_coverage_()
        coverages_2.append(coverage_2)
    while self.next_token_ != 'END_ADJUST':
        id_1 = self.expect_number_()
        id_2 = self.expect_number_()
        self.expect_keyword_('BY')
        pos_1 = self.parse_pos_()
        pos_2 = self.parse_pos_()
        adjust_pair[id_1, id_2] = (pos_1, pos_2)
    self.expect_keyword_('END_ADJUST')
    position = ast.PositionAdjustPairDefinition(coverages_1, coverages_2, adjust_pair, location=location)
    return position