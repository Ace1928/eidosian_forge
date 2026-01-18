import unicodedata
from pyglet.gl import *
from pyglet import image
def create_glyph(self, image, fmt=None):
    """Create a glyph using the given image.

        This is used internally by `Font` subclasses to add glyph data
        to the font.  Glyphs are packed within large textures maintained by
        `Font`.  This method inserts the image into a font texture and returns
        a glyph reference; it is up to the subclass to add metadata to the
        glyph.

        Applications should not use this method directly.

        :Parameters:
            `image` : `pyglet.image.AbstractImage`
                The image to write to the font texture.
            `fmt` : `int`
                Override for the format and internalformat of the atlas texture

        :rtype: `Glyph`
        """
    if self.texture_bin is None:
        if self.optimize_fit:
            self.texture_width, self.texture_height = self._get_optimal_atlas_size(image)
        self.texture_bin = GlyphTextureBin(self.texture_width, self.texture_height)
    glyph = self.texture_bin.add(image, fmt or self.texture_internalformat, self.texture_min_filter, self.texture_mag_filter, border=1)
    return glyph