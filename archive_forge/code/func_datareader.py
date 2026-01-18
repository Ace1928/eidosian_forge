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
def datareader(url, unquote=unquote):
    """Use "data" URL."""
    try:
        typ, data = url.split(',', 1)
    except ValueError:
        raise IOError('data error', 'bad data URL')
    if not typ:
        typ = 'text/plain;charset=US-ASCII'
    semi = typ.rfind(';')
    if semi >= 0 and '=' not in typ[semi:]:
        encoding = typ[semi + 1:]
        typ = typ[:semi]
    else:
        encoding = ''
    if encoding == 'base64':
        data = base64_decodebytes(data.encode('ascii'))
    else:
        data = unquote(data).encode('latin-1')
    return data