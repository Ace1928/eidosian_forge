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
def ll_fz_convert_pixmap(pix, cs_des, prf, default_cs, color_params, keep_alpha):
    """
    Low-level wrapper for `::fz_convert_pixmap()`.
    Convert an existing pixmap to a desired
    colorspace. Other properties of the pixmap, such as resolution
    and position are copied to the converted pixmap.

    pix: The pixmap to convert.

    default_cs: If NULL pix->colorspace is used. It is possible that
    the data may need to be interpreted as one of the color spaces
    in default_cs.

    cs_des: Desired colorspace, may be NULL to denote alpha-only.

    prf: Proofing color space through which we need to convert.

    color_params: Parameters that may be used in conversion (e.g.
    ri).

    keep_alpha: If 0 any alpha component is removed, otherwise
    alpha is kept if present in the pixmap.
    """
    return _mupdf.ll_fz_convert_pixmap(pix, cs_des, prf, default_cs, color_params, keep_alpha)