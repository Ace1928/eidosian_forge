from __future__ import annotations
from contextlib import redirect_stdout
import collections
import dataclasses
import json
import os
from pathlib import Path, PurePath
import sys
import typing as T
from . import build, mesonlib, coredata as cdata
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstJSONPrinter
from .backend import backends
from .dependencies import Dependency
from . import environment
from .interpreterbase import ObjectHolder
from .mesonlib import OptionKey
from .mparser import FunctionNode, ArrayNode, ArgumentNode, BaseStringNode
def nodes_to_paths(node_list: T.List[BaseNode]) -> T.List[Path]:
    res: T.List[Path] = []
    for n in node_list:
        args: T.List[BaseNode] = []
        if isinstance(n, FunctionNode):
            args = list(n.args.arguments)
            if n.func_name.value in BUILD_TARGET_FUNCTIONS:
                args.pop(0)
        elif isinstance(n, ArrayNode):
            args = n.args.arguments
        elif isinstance(n, ArgumentNode):
            args = n.arguments
        for j in args:
            if isinstance(j, BaseStringNode):
                assert isinstance(j.value, str)
                res += [Path(j.value)]
            elif isinstance(j, str):
                res += [Path(j)]
    res = [root_dir / i['subdir'] / x for x in res]
    res = [x.resolve() for x in res]
    return res