from __future__ import annotations
from .. import environment, mparser, mesonlib
from .baseobjects import (
from .exceptions import (
from .decorators import FeatureNew
from .disabler import Disabler, is_disabled
from .helpers import default_resolve_key, flatten, resolve_second_level_holders, stringifyUserArguments
from .operator import MesonOperator
from ._unholder import _unholder
import os, copy, re, pathlib
import typing as T
import textwrap
def sanity_check_ast(self) -> None:

    def _is_project(ast: mparser.CodeBlockNode) -> object:
        if not isinstance(ast, mparser.CodeBlockNode):
            raise InvalidCode('AST is of invalid type. Possibly a bug in the parser.')
        if not ast.lines:
            raise InvalidCode('No statements in code.')
        first = ast.lines[0]
        return isinstance(first, mparser.FunctionNode) and first.func_name.value == 'project'
    if not _is_project(self.ast):
        p = pathlib.Path(self.source_root).resolve()
        found = p
        for parent in p.parents:
            if (parent / 'meson.build').is_file():
                with open(parent / 'meson.build', encoding='utf-8') as f:
                    code = f.read()
                try:
                    ast = mparser.Parser(code, 'empty').parse()
                except mparser.ParseException:
                    continue
                if _is_project(ast):
                    found = parent
                    break
            else:
                break
        error = 'first statement must be a call to project()'
        if found != p:
            raise InvalidCode(f'Not the project root: {error}\n\nDid you mean to run meson from the directory: "{found}"?')
        else:
            raise InvalidCode(f'Invalid source tree: {error}')