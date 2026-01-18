import sys
from array import array
from functools import partial
from io import BytesIO
from . import Context, ImageSurface, constants, dlopen
from .ffi import ffi_pixbuf as ffi
def pixbuf_to_cairo_png(pixbuf):
    """Convert from PixBuf to ImageSurface, by going through the PNG format.

    This method is 10~30x slower than GDK but always works.

    """
    buffer_pointer = ffi.new('gchar **')
    buffer_size = ffi.new('gsize *')
    error = ffi.new('GError **')
    handle_g_error(error, pixbuf.save_to_buffer(buffer_pointer, buffer_size, ffi.new('char[]', b'png'), error, ffi.new('char[]', b'compression'), ffi.new('char[]', b'0'), ffi.NULL))
    png_bytes = ffi.buffer(buffer_pointer[0], buffer_size[0])
    return ImageSurface.create_from_png(BytesIO(png_bytes))