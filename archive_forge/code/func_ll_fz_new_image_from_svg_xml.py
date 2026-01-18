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
def ll_fz_new_image_from_svg_xml(xmldoc, xml, base_uri, dir):
    """
    Low-level wrapper for `::fz_new_image_from_svg_xml()`.
    Create a scalable image from an SVG document.
    """
    return _mupdf.ll_fz_new_image_from_svg_xml(xmldoc, xml, base_uri, dir)