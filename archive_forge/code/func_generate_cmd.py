from __future__ import annotations
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstPrinter
from mesonbuild.mesonlib import MesonException, setup_vsenv
from . import mlog, environment
from functools import wraps
from .mparser import Token, ArrayNode, ArgumentNode, AssignmentNode, BaseStringNode, BooleanNode, ElementaryNode, IdNode, FunctionNode, StringNode, SymbolNode
import json, os, re, sys
import typing as T
def generate_cmd(options) -> T.List[dict]:
    if os.path.exists(options.json):
        with open(options.json, encoding='utf-8') as fp:
            return json.load(fp)
    else:
        return json.loads(options.json)