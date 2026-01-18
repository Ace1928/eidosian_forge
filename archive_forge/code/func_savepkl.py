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
def savepkl(filename, record, versioning=False):
    from io import BytesIO
    with BytesIO() as f:
        if versioning:
            metadata = json.dumps({'version': version})
            f.write(metadata.encode('utf-8'))
            f.write('\n'.encode('utf-8'))
        pickle.dump(record, f)
        content = f.getvalue()
    pkl_open = gzip.open if filename.endswith('.pklz') else open
    tmpfile = filename + '.tmp'
    with pkl_open(tmpfile, 'wb') as pkl_file:
        pkl_file.write(content)
    for _ in range(5):
        try:
            os.rename(tmpfile, filename)
            break
        except FileNotFoundError as e:
            fmlogger.debug(str(e))
            sleep(2)
    else:
        raise e