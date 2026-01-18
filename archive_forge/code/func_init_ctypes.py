import os
import re
import sys
from typing import Any, Dict, List
from sphinx.errors import ExtensionError, SphinxError
from sphinx.search import SearchLanguage
from sphinx.util import import_object
def init_ctypes(self, options: Dict) -> None:
    import ctypes.util
    lib = options.get('lib')
    if lib is None:
        if sys.platform.startswith('win'):
            libname = 'libmecab.dll'
        else:
            libname = 'mecab'
        libpath = ctypes.util.find_library(libname)
    elif os.path.basename(lib) == lib:
        libpath = ctypes.util.find_library(lib)
    else:
        libpath = None
        if os.path.exists(lib):
            libpath = lib
    if libpath is None:
        raise RuntimeError('MeCab dynamic library is not available')
    param = 'mecab -Owakati'
    dict = options.get('dict')
    if dict:
        param += ' -d %s' % dict
    fs_enc = sys.getfilesystemencoding() or sys.getdefaultencoding()
    self.ctypes_libmecab = ctypes.CDLL(libpath)
    self.ctypes_libmecab.mecab_new2.argtypes = (ctypes.c_char_p,)
    self.ctypes_libmecab.mecab_new2.restype = ctypes.c_void_p
    self.ctypes_libmecab.mecab_sparse_tostr.argtypes = (ctypes.c_void_p, ctypes.c_char_p)
    self.ctypes_libmecab.mecab_sparse_tostr.restype = ctypes.c_char_p
    self.ctypes_mecab = self.ctypes_libmecab.mecab_new2(param.encode(fs_enc))
    if self.ctypes_mecab is None:
        raise SphinxError('mecab initialization failed')