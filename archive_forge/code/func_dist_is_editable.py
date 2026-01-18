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
def dist_is_editable(dist):
    """Is distribution an editable install?

    Parameters
    ----------
    dist : string
        Package name

    # Borrowed from `pip`'s' API
    """
    for path_item in sys.path:
        egg_link = op.join(path_item, dist + '.egg-link')
        if op.isfile(egg_link):
            return True
    return False