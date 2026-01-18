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
def ll_fz_new_display_list_from_svg_xml(xmldoc, xml, base_uri, dir):
    """
    Wrapper for out-params of fz_new_display_list_from_svg_xml().
    Returns: fz_display_list *, float w, float h
    """
    outparams = ll_fz_new_display_list_from_svg_xml_outparams()
    ret = ll_fz_new_display_list_from_svg_xml_outparams_fn(xmldoc, xml, base_uri, dir, outparams)
    return (ret, outparams.w, outparams.h)