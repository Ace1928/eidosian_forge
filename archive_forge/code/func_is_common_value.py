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
def is_common_value(a, b):
    a = unwrap_node(a)
    b = unwrap_node(b)
    if isinstance(a, ExprNodes.NameNode) and isinstance(b, ExprNodes.NameNode):
        return a.name == b.name
    if isinstance(a, ExprNodes.AttributeNode) and isinstance(b, ExprNodes.AttributeNode):
        return not a.is_py_attr and is_common_value(a.obj, b.obj) and (a.attribute == b.attribute)
    return False