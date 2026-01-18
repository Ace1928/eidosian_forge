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
def recursiveGetAttr(obj, name, g=None):
    """Can call down into e.g. object1.object2[4].attr"""
    if not isStr(name):
        raise TypeError('invalid recursive access of %s.%s' % (repr(obj), name))
    name = asNative(name)
    name = name.strip()
    if not name:
        raise ValueError('empty recursive access of %s' % repr(obj))
    dot = '.' if name and name[0] not in '[.(' else ''
    return rl_safe_eval('obj%s%s' % (dot, name), g={}, l=dict(obj=obj))