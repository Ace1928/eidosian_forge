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
def clean_working_directory(outputs, cwd, inputs, needed_outputs, config, files2keep=None, dirs2keep=None):
    """Removes all files not needed for further analysis from the directory"""
    if not outputs:
        return
    outputs_to_keep = list(outputs.trait_get().keys())
    if needed_outputs and str2bool(config['execution']['remove_unnecessary_outputs']):
        outputs_to_keep = needed_outputs
    output_files = []
    outputdict = outputs.trait_get()
    for output in outputs_to_keep:
        output_files.extend(walk_outputs(outputdict[output]))
    needed_files = [path for path, type in output_files if type == 'f']
    if str2bool(config['execution']['keep_inputs']):
        input_files = []
        inputdict = inputs.trait_get()
        input_files.extend(walk_outputs(inputdict))
        needed_files += [path for path, type in input_files if type == 'f']
    for extra in ['_0x*.json', 'provenance.*', 'pyscript*.m', 'pyjobs*.mat', 'command.txt', 'result*.pklz', '_inputs.pklz', '_node.pklz', '.proc-*']:
        needed_files.extend(glob(os.path.join(cwd, extra)))
    if files2keep:
        needed_files.extend(ensure_list(files2keep))
    needed_dirs = [path for path, type in output_files if type == 'd']
    if dirs2keep:
        needed_dirs.extend(ensure_list(dirs2keep))
    for extra in ['_nipype', '_report']:
        needed_dirs.extend(glob(os.path.join(cwd, extra)))
    temp = []
    for filename in needed_files:
        temp.extend(get_related_files(filename))
    needed_files = temp
    logger.debug('Needed files: %s', ';'.join(needed_files))
    logger.debug('Needed dirs: %s', ';'.join(needed_dirs))
    files2remove = []
    if str2bool(config['execution']['remove_unnecessary_outputs']):
        for f in walk_files(cwd):
            if f not in needed_files:
                if not needed_dirs:
                    files2remove.append(f)
                elif not any([f.startswith(dname) for dname in needed_dirs]):
                    files2remove.append(f)
    elif not str2bool(config['execution']['keep_inputs']):
        input_files = []
        inputdict = inputs.trait_get()
        input_files.extend(walk_outputs(inputdict))
        input_files = [path for path, type in input_files if type == 'f']
        for f in walk_files(cwd):
            if f in input_files and f not in needed_files:
                files2remove.append(f)
    logger.debug('Removing files: %s', ';'.join(files2remove))
    for f in files2remove:
        os.remove(f)
    for key in outputs.copyable_trait_names():
        if key not in outputs_to_keep:
            setattr(outputs, key, Undefined)
    return outputs