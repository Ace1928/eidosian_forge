import sys
from array import array
from functools import partial
from io import BytesIO
from . import Context, ImageSurface, constants, dlopen
from .ffi import ffi_pixbuf as ffi
def decode_to_pixbuf(image_data, width=None, height=None):
    """Decode an image from memory with GDK-PixBuf.
    The file format is detected automatically.

    :param image_data: A byte string
    :param width: Integer width in pixels or None
    :param height: Integer height in pixels or None
    :returns:
        A tuple of a new :class:`PixBuf` object
        and the name of the detected image format.
    :raises:
        :exc:`ImageLoadingError` if the image data is invalid
        or in an unsupported format.

    """
    loader = ffi.gc(gdk_pixbuf.gdk_pixbuf_loader_new(), gobject.g_object_unref)
    error = ffi.new('GError **')
    if width and height:
        gdk_pixbuf.gdk_pixbuf_loader_set_size(loader, width, height)
    handle_g_error(error, gdk_pixbuf.gdk_pixbuf_loader_write(loader, image_data, len(image_data), error))
    handle_g_error(error, gdk_pixbuf.gdk_pixbuf_loader_close(loader, error))
    format_ = gdk_pixbuf.gdk_pixbuf_loader_get_format(loader)
    format_name = ffi.string(gdk_pixbuf.gdk_pixbuf_format_get_name(format_)).decode('ascii') if format_ != ffi.NULL else None
    pixbuf = gdk_pixbuf.gdk_pixbuf_loader_get_pixbuf(loader)
    if pixbuf == ffi.NULL:
        raise ImageLoadingError('Not enough image data (got a NULL pixbuf.)')
    return (Pixbuf(pixbuf), format_name)