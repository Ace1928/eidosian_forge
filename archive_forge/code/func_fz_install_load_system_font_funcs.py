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
def fz_install_load_system_font_funcs(f=None, f_cjk=None, f_fallback=None):
    """
    Python override for MuPDF
    fz_install_load_system_font_funcs() using Swig Director
    support. Python callbacks are not passed a `ctx` arg, and
    can return None, a mupdf.fz_font or a mupdf.FzFont.
    """
    global g_fz_install_load_system_font_funcs_args
    g_fz_install_load_system_font_funcs_args = fz_install_load_system_font_funcs_args3(f, f_cjk, f_fallback)
    fz_install_load_system_font_funcs2(g_fz_install_load_system_font_funcs_args)