from __future__ import annotations
from . import Image
def query_palette(self, handle):
    """
        Installs the palette associated with the image in the given device
        context.

        This method should be called upon **QUERYNEWPALETTE** and
        **PALETTECHANGED** events from Windows. If this method returns a
        non-zero value, one or more display palette entries were changed, and
        the image should be redrawn.

        :param handle: Device context (HDC), cast to a Python integer, or an
                       HDC or HWND instance.
        :return: A true value if one or more entries were changed (this
                 indicates that the image should be redrawn).
        """
    if isinstance(handle, HWND):
        handle = self.image.getdc(handle)
        try:
            result = self.image.query_palette(handle)
        finally:
            self.image.releasedc(handle, handle)
    else:
        result = self.image.query_palette(handle)
    return result