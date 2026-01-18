import unicodedata
from pyglet.gl import *
from pyglet import image
def get_glyphs_for_width(self, text, width):
    """Return a list of glyphs for `text` that fit within the given width.
        
        If the entire text is larger than 'width', as much as possible will be
        used while breaking after a space or zero-width space character.  If a
        newline is encountered in text, only text up to that newline will be
        used.  If no break opportunities (newlines or spaces) occur within
        `width`, the text up to the first break opportunity will be used (this
        will exceed `width`).  If there are no break opportunities, the entire
        text will be used.

        You can assume that each character of the text is represented by
        exactly one glyph; so the amount of text "used up" can be determined
        by examining the length of the returned glyph list.

        :Parameters:
            `text` : str or unicode
                Text to render.
            `width` : int
                Maximum width of returned glyphs.
        
        :rtype: list of `Glyph`

        :see: `GlyphString`
        """
    glyph_renderer = None
    glyph_buffer = []
    glyphs = []
    for c in text:
        if c == '\n':
            glyphs += glyph_buffer
            break
        if c not in self.glyphs:
            if not glyph_renderer:
                glyph_renderer = self.glyph_renderer_class(self)
            self.glyphs[c] = glyph_renderer.render(c)
        glyph = self.glyphs[c]
        glyph_buffer.append(glyph)
        width -= glyph.advance
        if width <= 0 < len(glyphs):
            break
        if c in u' \u200b':
            glyphs += glyph_buffer
            glyph_buffer = []
    if len(glyphs) == 0:
        glyphs = glyph_buffer
    return glyphs