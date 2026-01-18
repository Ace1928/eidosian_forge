from itertools import groupby
import numpy as np
import param
import pyparsing as pp
from ..core.options import Cycle, Options, Palette
from ..core.util import merge_option_dicts
from ..operation import Compositor
from .transform import dim
@classmethod
def apply_deprecations(cls, path):
    """Convert any potentially deprecated paths and issue appropriate warnings"""
    split = path.split('.')
    msg = 'Element {old} deprecated. Use {new} instead.'
    for old, new in cls.deprecations:
        if split[0] == old:
            parsewarning.warning(msg.format(old=old, new=new))
            return '.'.join([new] + split[1:])
    return path