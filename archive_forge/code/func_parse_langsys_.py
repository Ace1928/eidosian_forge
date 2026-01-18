import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_langsys_(self):
    assert self.is_cur_keyword_('DEF_LANGSYS')
    location = self.cur_token_location_
    name = None
    if self.next_token_ == 'NAME':
        self.expect_keyword_('NAME')
        name = self.expect_string_()
    self.expect_keyword_('TAG')
    tag = self.expect_string_()
    features = []
    while self.next_token_ != 'END_LANGSYS':
        self.advance_lexer_()
        feature = self.parse_feature_()
        self.expect_keyword_('END_FEATURE')
        features.append(feature)
    def_langsys = ast.LangSysDefinition(name, tag, features, location=location)
    return def_langsys