from Xlib.protocol import request
from Xlib.xobject import resource, cursor
def create_glyph_cursor(self, mask, source_char, mask_char, f_rgb, b_rgb):
    fore_red, fore_green, fore_blue = f_rgb
    back_red, back_green, back_blue = b_rgb
    cid = self.display.allocate_resource_id()
    request.CreateGlyphCursor(display=self.display, cid=cid, source=self.id, mask=mask, source_char=source_char, mask_char=mask_char, fore_red=fore_red, fore_green=fore_green, fore_blue=fore_blue, back_red=back_red, back_green=back_green, back_blue=back_blue)
    cls = self.display.get_resource_class('cursor', cursor.Cursor)
    return cls(self.display, cid, owner=1)