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
def join_with_or(a, b, make_binop_node=ExprNodes.binop_node):
    or_node = make_binop_node(node.pos, 'or', a, b)
    or_node.type = PyrexTypes.c_bint_type
    or_node.wrap_operands(env)
    return or_node