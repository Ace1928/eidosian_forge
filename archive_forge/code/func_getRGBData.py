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
def getRGBData(self):
    """Return byte array of RGB data as string"""
    try:
        if self._data is None:
            self._dataA = None
            im = self._image
            mode = self.mode = im.mode
            if mode in ('LA', 'RGBA'):
                if getattr(Image, 'VERSION', '').startswith('1.1.7'):
                    im.load()
                self._dataA = ImageReader(im.split()[3 if mode == 'RGBA' else 1])
                nm = mode[:-1]
                im = im.convert(nm)
                self.mode = nm
            elif mode not in ('L', 'RGB', 'CMYK'):
                if im.format == 'PNG' and im.mode == 'P' and ('transparency' in im.info):
                    im = im.convert('RGBA')
                    self._dataA = ImageReader(im.split()[3])
                    im = im.convert('RGB')
                else:
                    im = im.convert('RGB')
                self.mode = 'RGB'
            self._data = (im.tobytes if hasattr(im, 'tobytes') else im.tostring)()
        return self._data
    except:
        annotateException('\nidentity=%s' % self.identity())