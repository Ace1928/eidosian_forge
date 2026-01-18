import sys
from array import array
from functools import partial
from io import BytesIO
from . import Context, ImageSurface, constants, dlopen
from .ffi import ffi_pixbuf as ffi
def pixbuf_to_cairo_gdk(pixbuf):
    """Convert from PixBuf to ImageSurface, using GDK.

    This method is fastest but GDK is not always available.

    """
    dummy_context = Context(ImageSurface(constants.FORMAT_ARGB32, 1, 1))
    gdk.gdk_cairo_set_source_pixbuf(dummy_context._pointer, pixbuf._pointer, 0, 0)
    return dummy_context.get_source().get_surface()