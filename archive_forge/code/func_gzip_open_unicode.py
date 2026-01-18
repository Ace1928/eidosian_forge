import codecs
import functools
import os
import pickle
import re
import sys
import textwrap
import zipfile
from abc import ABCMeta, abstractmethod
from gzip import WRITE as GZ_WRITE
from gzip import GzipFile
from io import BytesIO, TextIOWrapper
from urllib.request import url2pathname, urlopen
from nltk import grammar, sem
from nltk.compat import add_py3_data, py3_data
from nltk.internals import deprecated
def gzip_open_unicode(filename, mode='rb', compresslevel=9, encoding='utf-8', fileobj=None, errors=None, newline=None):
    if fileobj is None:
        fileobj = GzipFile(filename, mode, compresslevel, fileobj)
    return TextIOWrapper(fileobj, encoding, errors, newline)