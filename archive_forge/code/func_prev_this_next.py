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
def prev_this_next(items):
    """
    Loop over a collection with look-ahead and look-back.
    
    From Thomas Guest, 
    http://wordaligned.org/articles/zippy-triples-served-with-python
    
    Seriously useful looping tool (Google "zippy triples")
    lets you loop a collection and see the previous and next items,
    which get set to None at the ends.
    
    To be used in layout algorithms where one wants a peek at the
    next item coming down the pipe.

    """
    extend = itertools.chain([None], items, [None])
    prev, this, next = itertools.tee(extend, 3)
    try:
        next(this)
        next(next)
        next(next)
    except StopIteration:
        pass
    return zip(prev, this, next)