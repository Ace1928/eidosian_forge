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
def commasplit(s):
    """
    Splits the string s at every unescaped comma and returns the result as a list.
    To escape a comma, double it. Individual items are stripped.
    To avoid the ambiguity of 3 successive commas to denote a comma at the beginning
    or end of an item, add a space between the item seperator and the escaped comma.
    
    >>> commasplit(u'a,b,c') == [u'a', u'b', u'c']
    True
    >>> commasplit('a,, , b , c    ') == [u'a,', u'b', u'c']
    True
    >>> commasplit(u'a, ,,b, c') == [u'a', u',b', u'c']
    """
    if isBytes(s):
        s = s.decode('utf8')
    n = len(s) - 1
    s += u' '
    i = 0
    r = [u'']
    while i <= n:
        if s[i] == u',':
            if s[i + 1] == u',':
                r[-1] += u','
                i += 1
            else:
                r[-1] = r[-1].strip()
                if i != n:
                    r.append(u'')
        else:
            r[-1] += s[i]
        i += 1
    r[-1] = r[-1].strip()
    return r