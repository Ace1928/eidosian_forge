from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_new_image_from_compressed_buffer2(w, h, bpc, colorspace, xres, yres, interpolate, imagemask, decode, colorkey, buffer, mask):
    """
     Low-level wrapper for `::fz_new_image_from_compressed_buffer2()`.  Swig-friendly wrapper for fz_new_image_from_compressed_buffer(),
    uses specified `decode` and `colorkey` if they are not null (in which
    case we assert that they have size `2*fz_colorspace_n(colorspace)`).
    """
    return _mupdf.ll_fz_new_image_from_compressed_buffer2(w, h, bpc, colorspace, xres, yres, interpolate, imagemask, decode, colorkey, buffer, mask)