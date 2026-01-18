import os, pickle, sys, time, types, datetime, importlib
from ast import literal_eval
from base64 import decodebytes as base64_decodebytes, encodebytes as base64_encodebytes
from io import BytesIO
from hashlib import md5
from reportlab.lib.rltempfile import get_rl_tempfile, get_rl_tempdir
from . rl_safe_eval import rl_safe_exec, rl_safe_eval, safer_globals, rl_extended_literal_eval
from PIL import Image
import builtins
import reportlab
import glob, fnmatch
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
from importlib import util as importlib_util
import itertools
def rl_get_module(name, dir):
    if name in sys.modules:
        om = sys.modules[name]
        del sys.modules[name]
    else:
        om = None
    try:
        try:
            return __rl_get_module__(name, dir)
        except:
            if isCompactDistro():
                import zipimport
                dir = _startswith_rl(dir)
                dir = (dir == '.' or not dir) and _archive or os.path.join(_archive, dir.replace('/', os.sep))
                zi = zipimport.zipimporter(dir)
                return zi.load_module(name)
            raise ImportError('%s[%s]' % (name, dir))
    finally:
        if om:
            sys.modules[name] = om