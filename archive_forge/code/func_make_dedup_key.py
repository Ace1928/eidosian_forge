from __future__ import absolute_import
import cython
import re
import sys
import copy
import os.path
import operator
from .Errors import (
from .Code import UtilityCode, TempitaUtilityCode
from . import StringEncoding
from . import Naming
from . import Nodes
from .Nodes import Node, utility_code_for_imports, SingleAssignmentNode
from . import PyrexTypes
from .PyrexTypes import py_object_type, typecast, error_type, \
from . import TypeSlots
from .Builtin import (
from . import Builtin
from . import Symtab
from .. import Utils
from .Annotate import AnnotationItem
from . import Future
from ..Debugging import print_call_chain
from .DebugFlags import debug_disposal_code, debug_coercion
from .Pythran import (to_pythran, is_pythran_supported_type, is_pythran_supported_operation_type,
from .PyrexTypes import PythranExpr
def make_dedup_key(outer_type, item_nodes):
    """
    Recursively generate a deduplication key from a sequence of values.
    Includes Cython node types to work around the fact that (1, 2.0) == (1.0, 2), for example.

    @param outer_type: The type of the outer container.
    @param item_nodes: A sequence of constant nodes that will be traversed recursively.
    @return: A tuple that can be used as a dict key for deduplication.
    """
    item_keys = [(py_object_type, None, type(None)) if node is None else make_dedup_key(node.type, [node.mult_factor if node.is_literal else None] + node.args) if node.is_sequence_constructor else make_dedup_key(node.type, (node.start, node.stop, node.step)) if node.is_slice else (node.type, node.constant_result, type(node.constant_result) if node.type is py_object_type else None) if node.has_constant_result() else (node.type, node.value, node.unicode_value, 'IdentifierStringNode') if isinstance(node, IdentifierStringNode) else None for node in item_nodes]
    if None in item_keys:
        return None
    return (outer_type, tuple(item_keys))