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
def show_cfg(resource_url, escape='##'):
    """
    Write out a grammar file, ignoring escaped and empty lines.

    :type resource_url: str
    :param resource_url: A URL specifying where the resource should be
        loaded from.  The default protocol is "nltk:", which searches
        for the file in the the NLTK data package.
    :type escape: str
    :param escape: Prepended string that signals lines to be ignored
    """
    resource_url = normalize_resource_url(resource_url)
    resource_val = load(resource_url, format='text', cache=False)
    lines = resource_val.splitlines()
    for l in lines:
        if l.startswith(escape):
            continue
        if re.match('^$', l):
            continue
        print(l)