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
def findInPaths(fn, paths, isfile=True, fail=False):
    """search for relative files in likely places"""
    exists = isfile and os.path.isfile or os.path.isdir
    if exists(fn):
        return fn
    pjoin = os.path.join
    if not os.path.isabs(fn):
        for p in paths:
            pfn = pjoin(p, fn)
            if exists(pfn):
                return pfn
    if fail:
        raise ValueError('cannot locate %r with paths=%r' % (fn, paths))
    return fn