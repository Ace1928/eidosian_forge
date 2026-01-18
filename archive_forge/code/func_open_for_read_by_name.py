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
def open_for_read_by_name(name, mode='b'):
    if 'r' not in mode:
        mode = 'r' + mode
    try:
        return open(name, mode)
    except IOError:
        if _isFSD or __rl_loader__ is None:
            raise
        name = _startswith_rl(name)
        s = __rl_loader__.get_data(name)
        if 'b' not in mode and os.linesep != '\n':
            s = s.replace(os.linesep, '\n')
        return BytesIO(s)