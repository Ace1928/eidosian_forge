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
def recursiveImport(modulename, baseDir=None, noCWD=0, debug=0):
    """Dynamically imports possible packagized module, or raises ImportError"""
    path = [normalize_path(p) for p in sys.path]
    if baseDir:
        for p in baseDir if isinstance(baseDir, (list, tuple)) else (baseDir,):
            if p:
                p = normalize_path(p)
                if p not in path:
                    path.insert(0, p)
    if noCWD:
        for p in ('', '.', normalize_path('.')):
            while p in path:
                if debug:
                    print('removed "%s" from path' % p)
                path.remove(p)
    else:
        p = os.getcwd()
        if p not in path:
            path.insert(0, p)
    opath = sys.path
    try:
        sys.path = path
        _importlib_invalidate_caches()
        if debug:
            print()
            print(20 * '+')
            print('+++++ modulename=%s' % ascii(modulename))
            print('+++++ cwd=%s' % ascii(os.getcwd()))
            print('+++++ sys.path=%s' % ascii(sys.path))
            print('+++++ os.paths.isfile(%s)=%s' % (ascii('./%s.py' % modulename), ascii(os.path.isfile('./%s.py' % modulename))))
            print('+++++ opath=%s' % ascii(opath))
            print(20 * '-')
        return importlib.import_module(modulename)
    except ImportError:
        annotateException('Could not import %r\nusing sys.path %r in cwd=%r' % (modulename, sys.path, os.getcwd()))
    except:
        annotateException('Exception %s while importing %r\nusing sys.path %r in cwd=%r' % (str(sys.exc_info()[1]), modulename, sys.path, os.getcwd()))
    finally:
        sys.path = opath
        _importlib_invalidate_caches()
        if debug:
            print('===== restore sys.path=%s' % repr(opath))