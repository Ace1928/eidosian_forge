from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
def find_assignment_node(self, node: BaseNode) -> AssignmentNode:
    if node.ast_id and node.ast_id in self.interpreter.reverse_assignment:
        return self.interpreter.reverse_assignment[node.ast_id]
    return None