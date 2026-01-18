import sys
import pickle
import errno
import subprocess as sp
import gzip
import hashlib
import locale
from hashlib import md5
import os
import os.path as op
import re
import shutil
import contextlib
import posixpath
from pathlib import Path
import simplejson as json
from time import sleep, time
from .. import logging, config, __version__ as version
from .misc import is_container
def loadcrash(infile, *args):
    if infile.endswith('pkl') or infile.endswith('pklz'):
        return loadpkl(infile)
    else:
        raise ValueError('Only pickled crashfiles are supported')