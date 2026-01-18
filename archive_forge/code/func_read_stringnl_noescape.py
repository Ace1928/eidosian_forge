import codecs
import io
import pickle
import re
import sys
from struct import unpack as _unpack
from pickle import decode_long
def read_stringnl_noescape(f):
    return read_stringnl(f, stripquotes=False)