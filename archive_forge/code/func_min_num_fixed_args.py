from __future__ import absolute_import
from . import Naming
from . import PyrexTypes
from .Errors import error
import copy
def min_num_fixed_args(self):
    return self.max_num_fixed_args() - self.optional_object_arg_count