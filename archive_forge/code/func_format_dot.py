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
def format_dot(dotfilename, format='png'):
    """Dump a directed graph (Linux only; install via `brew` on OSX)"""
    try:
        formatted_dot, _ = _run_dot(dotfilename, format_ext=format)
    except IOError as ioe:
        if 'could not be found' in str(ioe):
            raise IOError("Cannot draw directed graph; executable 'dot' is unavailable")
        else:
            raise ioe
    return formatted_dot