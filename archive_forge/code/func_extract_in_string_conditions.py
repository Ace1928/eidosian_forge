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
def extract_in_string_conditions(self, string_literal):
    if isinstance(string_literal, ExprNodes.UnicodeNode):
        charvals = list(map(ord, set(string_literal.value)))
        charvals.sort()
        return [ExprNodes.IntNode(string_literal.pos, value=str(charval), constant_result=charval) for charval in charvals]
    else:
        characters = string_literal.value
        characters = list({characters[i:i + 1] for i in range(len(characters))})
        characters.sort()
        return [ExprNodes.CharNode(string_literal.pos, value=charval, constant_result=charval) for charval in characters]