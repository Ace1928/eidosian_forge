import os
import sys
import pickle
from collections import defaultdict
import re
from copy import deepcopy
from glob import glob
from pathlib import Path
from traceback import format_exception
from hashlib import sha1
from functools import reduce
import numpy as np
from ... import logging, config
from ...utils.filemanip import (
from ...utils.misc import str2bool
from ...utils.functions import create_function_from_source
from ...interfaces.base.traits_extension import (
from ...interfaces.base.support import Bunch, InterfaceResult
from ...interfaces.base import CommandLine
from ...interfaces.utility import IdentityInterface
from ...utils.provenance import ProvStore, pm, nipype_ns, get_id
from inspect import signature
def modify_paths(object, relative=True, basedir=None):
    """Convert paths in data structure to either full paths or relative paths

    Supports combinations of lists, dicts, tuples, strs

    Parameters
    ----------

    relative : boolean indicating whether paths should be set relative to the
               current directory
    basedir : default os.getcwd()
              what base directory to use as default
    """
    if not basedir:
        basedir = os.getcwd()
    if isinstance(object, dict):
        out = {}
        for key, val in sorted(object.items()):
            if isdefined(val):
                out[key] = modify_paths(val, relative=relative, basedir=basedir)
    elif isinstance(object, (list, tuple)):
        out = []
        for val in object:
            if isdefined(val):
                out.append(modify_paths(val, relative=relative, basedir=basedir))
        if isinstance(object, tuple):
            out = tuple(out)
    elif isdefined(object):
        if isinstance(object, (str, bytes)) and os.path.isfile(object):
            if relative:
                if config.getboolean('execution', 'use_relative_paths'):
                    out = relpath(object, start=basedir)
                else:
                    out = object
            else:
                out = os.path.abspath(os.path.join(basedir, object))
            if not os.path.exists(out):
                raise IOError('File %s not found' % out)
        else:
            out = object
    else:
        raise TypeError('Object {} is undefined'.format(object))
    return out