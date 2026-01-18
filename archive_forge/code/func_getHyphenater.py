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
def getHyphenater(hDict=None):
    try:
        from reportlab.lib.pyHnj import Hyphen
        if hDict is None:
            hDict = os.path.join(os.path.dirname(__file__), 'hyphen.mashed')
        return Hyphen(hDict)
    except ImportError as errMsg:
        if str(errMsg) != 'No module named pyHnj':
            raise
        return None