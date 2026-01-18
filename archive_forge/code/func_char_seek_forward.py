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
def char_seek_forward(self, offset):
    """
        Move the read pointer forward by ``offset`` characters.
        """
    if offset < 0:
        raise ValueError('Negative offsets are not supported')
    self.seek(self.tell())
    self._char_seek_forward(offset)