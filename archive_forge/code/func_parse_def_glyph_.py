import fontTools.voltLib.ast as ast
from fontTools.voltLib.lexer import Lexer
from fontTools.voltLib.error import VoltLibError
from io import open
def parse_def_glyph_(self):
    assert self.is_cur_keyword_('DEF_GLYPH')
    location = self.cur_token_location_
    name = self.expect_string_()
    self.expect_keyword_('ID')
    gid = self.expect_number_()
    if gid < 0:
        raise VoltLibError('Invalid glyph ID', self.cur_token_location_)
    gunicode = None
    if self.next_token_ == 'UNICODE':
        self.expect_keyword_('UNICODE')
        gunicode = [self.expect_number_()]
        if gunicode[0] < 0:
            raise VoltLibError('Invalid glyph UNICODE', self.cur_token_location_)
    elif self.next_token_ == 'UNICODEVALUES':
        self.expect_keyword_('UNICODEVALUES')
        gunicode = self.parse_unicode_values_()
    gtype = None
    if self.next_token_ == 'TYPE':
        self.expect_keyword_('TYPE')
        gtype = self.expect_name_()
        assert gtype in ('BASE', 'LIGATURE', 'MARK', 'COMPONENT')
    components = None
    if self.next_token_ == 'COMPONENTS':
        self.expect_keyword_('COMPONENTS')
        components = self.expect_number_()
    self.expect_keyword_('END_GLYPH')
    if self.glyphs_.resolve(name) is not None:
        raise VoltLibError('Glyph "%s" (gid %i) already defined' % (name, gid), location)
    def_glyph = ast.GlyphDefinition(name, gid, gunicode, gtype, components, location=location)
    self.glyphs_.define(name, def_glyph)
    return def_glyph