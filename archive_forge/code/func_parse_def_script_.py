import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_def_script_(self):
    assert self.is_cur_keyword_('DEF_SCRIPT')
    location = self.cur_token_location_
    name = None
    if self.next_token_ == 'NAME':
        self.expect_keyword_('NAME')
        name = self.expect_string_()
    self.expect_keyword_('TAG')
    tag = self.expect_string_()
    if self.scripts_.resolve(tag) is not None:
        raise VoltLibError('Script "%s" already defined, script tags are case insensitive' % tag, location)
    self.langs_.enter_scope()
    langs = []
    while self.next_token_ != 'END_SCRIPT':
        self.advance_lexer_()
        lang = self.parse_langsys_()
        self.expect_keyword_('END_LANGSYS')
        if self.langs_.resolve(lang.tag) is not None:
            raise VoltLibError('Language "%s" already defined in script "%s", language tags are case insensitive' % (lang.tag, tag), location)
        self.langs_.define(lang.tag, lang)
        langs.append(lang)
    self.expect_keyword_('END_SCRIPT')
    self.langs_.exit_scope()
    def_script = ast.ScriptDefinition(name, tag, langs, location=location)
    self.scripts_.define(tag, def_script)
    return def_script