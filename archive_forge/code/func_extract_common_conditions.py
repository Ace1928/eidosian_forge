from __future__ import absolute_import
import re
import sys
import copy
import codecs
import itertools
from . import TypeSlots
from .ExprNodes import not_a_constant
import cython
from . import Nodes
from . import ExprNodes
from . import PyrexTypes
from . import Visitor
from . import Builtin
from . import UtilNodes
from . import Options
from .Code import UtilityCode, TempitaUtilityCode
from .StringEncoding import EncodedString, bytes_literal, encoded_string
from .Errors import error, warning
from .ParseTreeTransforms import SkipDeclarations
from .. import Utils
def extract_common_conditions(self, common_var, condition, allow_not_in):
    not_in, var, conditions = self.extract_conditions(condition, allow_not_in)
    if var is None:
        return self.NO_MATCH
    elif common_var is not None and (not is_common_value(var, common_var)):
        return self.NO_MATCH
    elif not (var.type.is_int or var.type.is_enum) or any([not (cond.type.is_int or cond.type.is_enum) for cond in conditions]):
        return self.NO_MATCH
    return (not_in, var, conditions)