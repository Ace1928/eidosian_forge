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
def crash2txt(filename, record):
    """Write out plain text crash file"""
    with open(filename, 'w') as fp:
        if 'node' in record:
            node = record['node']
            fp.write('Node: {}\n'.format(node.fullname))
            fp.write('Working directory: {}\n'.format(node.output_dir()))
            fp.write('\n')
            fp.write('Node inputs:\n{}\n'.format(node.inputs))
        fp.write(''.join(record['traceback']))