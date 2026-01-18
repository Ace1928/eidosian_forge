import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_feature_(self):
    assert self.is_cur_keyword_('DEF_FEATURE')
    location = self.cur_token_location_
    self.expect_keyword_('NAME')
    name = self.expect_string_()
    self.expect_keyword_('TAG')
    tag = self.expect_string_()
    lookups = []
    while self.next_token_ != 'END_FEATURE':
        self.expect_keyword_('LOOKUP')
        lookup = self.expect_string_()
        lookups.append(lookup)
    feature = ast.FeatureDefinition(name, tag, lookups, location=location)
    return feature