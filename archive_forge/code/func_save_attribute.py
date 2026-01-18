from pythran.tables import MODULES
from pythran.intrinsic import Class
from pythran.typing import Tuple, List, Set, Dict
from pythran.utils import isstr
from pythran import metadata
import beniget
import gast as ast
import logging
import numpy as np
def save_attribute(module):
    """ Recursively save Pythonic keywords as possible attributes. """
    self.attributes.update(module.keys())
    for signature in module.values():
        if isinstance(signature, dict):
            save_attribute(signature)
        elif isinstance(signature, Class):
            save_attribute(signature.fields)