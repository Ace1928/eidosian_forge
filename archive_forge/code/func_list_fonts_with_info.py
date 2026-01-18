import types
from . import error, ext, X
from Xlib.protocol import display, request, event, rq
import Xlib.xobject.resource
import Xlib.xobject.drawable
import Xlib.xobject.fontable
import Xlib.xobject.colormap
import Xlib.xobject.cursor
def list_fonts_with_info(self, pattern, max_names):
    """Return a list of fonts matching pattern. No more than
        max_names will be returned. Each list item represents one font
        and has the following properties:

        name
            The name of the font.
        min_bounds
        max_bounds
        min_char_or_byte2
        max_char_or_byte2
        default_char
        draw_direction
        min_byte1
        max_byte1
        all_chars_exist
        font_ascent
        font_descent
        replies_hint
            See the descripton of XFontStruct in XGetFontProperty(3X11)
            for details on these values.
        properties
            A list of properties. Each entry has two attributes:

            name
                The atom identifying this property.
            value
                A 32-bit unsigned value.
        """
    return request.ListFontsWithInfo(display=self.display, max_names=max_names, pattern=pattern)