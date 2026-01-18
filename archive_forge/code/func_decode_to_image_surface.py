import sys
from array import array
from functools import partial
from io import BytesIO
from . import Context, ImageSurface, constants, dlopen
from .ffi import ffi_pixbuf as ffi
def decode_to_image_surface(image_data, width=None, height=None):
    """Decode an image from memory into a cairo surface.
    The file format is detected automatically.

    :param image_data: A byte string
    :param width: Integer width in pixels or None
    :param height: Integer height in pixels or None
    :returns:
        A tuple of a new :class:`~cairocffi.ImageSurface` object
        and the name of the detected image format.
    :raises:
        :exc:`ImageLoadingError` if the image data is invalid
        or in an unsupported format.

    """
    pixbuf, format_name = decode_to_pixbuf(image_data, width, height)
    surface = pixbuf_to_cairo_gdk(pixbuf) if gdk is not None else pixbuf_to_cairo_slices(pixbuf) if not pixbuf.get_has_alpha() else pixbuf_to_cairo_png(pixbuf)
    return (surface, format_name)