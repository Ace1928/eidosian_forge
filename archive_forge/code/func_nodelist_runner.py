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
def nodelist_runner(nodes, updatehash=False, stop_first=False):
    """
    A generator that iterates and over a list of ``nodes`` and
    executes them.

    """
    for i, node in nodes:
        err = None
        result = None
        try:
            result = node.run(updatehash=updatehash)
        except Exception:
            if stop_first:
                raise
            result = node.result
            err = []
            if result.runtime and hasattr(result.runtime, 'traceback'):
                err = [result.runtime.traceback]
            err += format_exception(*sys.exc_info())
            err = '\n'.join(err)
        finally:
            yield (i, result, err)