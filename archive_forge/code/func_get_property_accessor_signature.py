from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def get_property_accessor_signature(name):
    return property_accessor_signatures.get(name)