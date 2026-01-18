import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_enum_(self):
    self.expect_keyword_('ENUM')
    location = self.cur_token_location_
    enum = ast.Enum(self.parse_coverage_(), location=location)
    self.expect_keyword_('END_ENUM')
    return enum