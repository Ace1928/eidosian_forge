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
def hash_timestamp(afile):
    """Computes md5 hash of the timestamp of a file"""
    md5hex = None
    if op.isfile(afile):
        md5obj = md5()
        stat = os.stat(afile)
        md5obj.update(str(stat.st_size).encode())
        md5obj.update(str(stat.st_mtime).encode())
        md5hex = md5obj.hexdigest()
    return md5hex