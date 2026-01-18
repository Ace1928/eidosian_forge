import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_coverage_(self):
    coverage = []
    location = self.cur_token_location_
    while self.next_token_ in ('GLYPH', 'GROUP', 'RANGE', 'ENUM'):
        if self.next_token_ == 'ENUM':
            enum = self.parse_enum_()
            coverage.append(enum)
        elif self.next_token_ == 'GLYPH':
            self.expect_keyword_('GLYPH')
            name = self.expect_string_()
            coverage.append(ast.GlyphName(name, location=location))
        elif self.next_token_ == 'GROUP':
            self.expect_keyword_('GROUP')
            name = self.expect_string_()
            coverage.append(ast.GroupName(name, self, location=location))
        elif self.next_token_ == 'RANGE':
            self.expect_keyword_('RANGE')
            start = self.expect_string_()
            self.expect_keyword_('TO')
            end = self.expect_string_()
            coverage.append(ast.Range(start, end, self, location=location))
    return tuple(coverage)