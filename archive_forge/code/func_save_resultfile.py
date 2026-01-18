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
def save_resultfile(result, cwd, name, rebase=None):
    """Save a result pklz file to ``cwd``."""
    if rebase is None:
        rebase = config.getboolean('execution', 'use_relative_paths')
    cwd = os.path.abspath(cwd)
    resultsfile = os.path.join(cwd, 'result_%s.pklz' % name)
    logger.debug("Saving results file: '%s'", resultsfile)
    if result.outputs is None:
        logger.warning('Storing result file without outputs')
        savepkl(resultsfile, result)
        return
    try:
        output_names = result.outputs.copyable_trait_names()
    except AttributeError:
        logger.debug('Storing non-traited results, skipping rebase of paths')
        savepkl(resultsfile, result)
        return
    if not rebase:
        savepkl(resultsfile, result)
        return
    backup_traits = {}
    try:
        with indirectory(cwd):
            for key in output_names:
                old = getattr(result.outputs, key)
                if isdefined(old):
                    if result.outputs.trait(key).is_trait_type(OutputMultiPath):
                        old = result.outputs.trait(key).handler.get_value(result.outputs, key)
                    backup_traits[key] = old
                    val = rebase_path_traits(result.outputs.trait(key), old, cwd)
                    setattr(result.outputs, key, val)
        savepkl(resultsfile, result)
    finally:
        for key, val in list(backup_traits.items()):
            setattr(result.outputs, key, val)