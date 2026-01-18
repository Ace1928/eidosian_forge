from __future__ import absolute_import
import cython
import copy
import hashlib
import sys
from . import PyrexTypes
from . import Naming
from . import ExprNodes
from . import Nodes
from . import Options
from . import Builtin
from . import Errors
from .Visitor import VisitorTransform, TreeVisitor
from .Visitor import CythonTransform, EnvTransform, ScopeTrackingTransform
from .UtilNodes import LetNode, LetRefNode
from .TreeFragment import TreeFragment
from .StringEncoding import EncodedString, _unicode
from .Errors import error, warning, CompileError, InternalError
from .Code import UtilityCode
def flatten_parallel_assignments(input, output):
    rhs = input[-1]
    if not (rhs.is_sequence_constructor or isinstance(rhs, ExprNodes.UnicodeNode)) or not sum([lhs.is_sequence_constructor for lhs in input[:-1]]):
        output.append(input)
        return
    complete_assignments = []
    if rhs.is_sequence_constructor:
        rhs_args = rhs.args
    elif rhs.is_string_literal:
        rhs_args = unpack_string_to_character_literals(rhs)
    rhs_size = len(rhs_args)
    lhs_targets = [[] for _ in range(rhs_size)]
    starred_assignments = []
    for lhs in input[:-1]:
        if not lhs.is_sequence_constructor:
            if lhs.is_starred:
                error(lhs.pos, 'starred assignment target must be in a list or tuple')
            complete_assignments.append(lhs)
            continue
        lhs_size = len(lhs.args)
        starred_targets = sum([1 for expr in lhs.args if expr.is_starred])
        if starred_targets > 1:
            error(lhs.pos, 'more than 1 starred expression in assignment')
            output.append([lhs, rhs])
            continue
        elif lhs_size - starred_targets > rhs_size:
            error(lhs.pos, 'need more than %d value%s to unpack' % (rhs_size, rhs_size != 1 and 's' or ''))
            output.append([lhs, rhs])
            continue
        elif starred_targets:
            map_starred_assignment(lhs_targets, starred_assignments, lhs.args, rhs_args)
        elif lhs_size < rhs_size:
            error(lhs.pos, 'too many values to unpack (expected %d, got %d)' % (lhs_size, rhs_size))
            output.append([lhs, rhs])
            continue
        else:
            for targets, expr in zip(lhs_targets, lhs.args):
                targets.append(expr)
    if complete_assignments:
        complete_assignments.append(rhs)
        output.append(complete_assignments)
    for cascade, rhs in zip(lhs_targets, rhs_args):
        if cascade:
            cascade.append(rhs)
            flatten_parallel_assignments(cascade, output)
    for cascade in starred_assignments:
        if cascade[0].is_sequence_constructor:
            flatten_parallel_assignments(cascade, output)
        else:
            output.append(cascade)