from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
def find_dependency(self, dependency: str):

    def check_list(name: str):
        for i in self.interpreter.dependencies:
            if name == i['name']:
                return i
        return None
    dep = check_list(dependency)
    if dep is not None:
        return dep
    if dependency in self.interpreter.assignments:
        node = self.interpreter.assignments[dependency]
        if isinstance(node, FunctionNode):
            if node.func_name.value == 'dependency':
                name = self.interpreter.flatten_args(node.args)[0]
                dep = check_list(name)
    return dep