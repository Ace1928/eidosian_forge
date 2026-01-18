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
def get_print_name(node, simple_form=True):
    """Get the name of the node

    For example, a node containing an instance of interfaces.fsl.BET
    would be called nodename.BET.fsl

    """
    name = node.fullname
    if hasattr(node, '_interface'):
        pkglist = node.interface.__class__.__module__.split('.')
        interface = node.interface.__class__.__name__
        destclass = ''
        if len(pkglist) > 2:
            destclass = '.%s' % pkglist[2]
        if simple_form:
            name = node.fullname + destclass
        else:
            name = '.'.join([node.fullname, interface]) + destclass
    if simple_form:
        parts = name.split('.')
        if len(parts) > 2:
            return ' ('.join(parts[1:]) + ')'
        elif len(parts) == 2:
            return parts[1]
    return name