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
def load_resultfile(results_file, resolve=True):
    """
    Load InterfaceResult file from path.

    Parameters
    ----------
    results_file : pathlike
        Path to an existing pickle (``result_<interface name>.pklz``) created with
        ``save_resultfile``.
        Raises ``FileNotFoundError`` if ``results_file`` does not exist.
    resolve : bool
        Determines whether relative paths will be resolved to absolute (default is ``True``).

    Returns
    -------
    result : InterfaceResult
        A Nipype object containing the runtime, inputs, outputs and other interface information
        such as a traceback in the case of errors.

    """
    results_file = Path(results_file)
    if not results_file.exists():
        raise FileNotFoundError(results_file)
    result = loadpkl(results_file)
    if resolve and getattr(result, 'outputs', None):
        try:
            outputs = result.outputs.get()
        except TypeError:
            logger.debug('Outputs object of loaded result %s is a Bunch.', results_file)
            return result
        logger.debug('Resolving paths in outputs loaded from results file.')
        for trait_name, old in list(outputs.items()):
            if isdefined(old):
                if result.outputs.trait(trait_name).is_trait_type(OutputMultiPath):
                    old = result.outputs.trait(trait_name).handler.get_value(result.outputs, trait_name)
                value = resolve_path_traits(result.outputs.trait(trait_name), old, results_file.parent)
                setattr(result.outputs, trait_name, value)
    return result