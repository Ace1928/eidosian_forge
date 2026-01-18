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
def canonicalize_env(env):
    """Windows requires that environment be dicts with str as keys and values
    This function converts any unicode entries for Windows only, returning the
    dictionary untouched in other environments.

    Parameters
    ----------
    env : dict
        environment dictionary with unicode or bytes keys and values

    Returns
    -------
    env : dict
        Windows: environment dictionary with str keys and values
        Other: untouched input ``env``
    """
    if os.name != 'nt':
        return env
    out_env = {}
    for key, val in env.items():
        if not isinstance(key, str):
            key = key.decode('utf-8')
        if not isinstance(val, str):
            val = val.decode('utf-8')
        out_env[key] = val
    return out_env