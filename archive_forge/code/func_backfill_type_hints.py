from __future__ import annotations
import inspect
import re
import sys
import textwrap
import types
from ast import FunctionDef, Module, stmt
from dataclasses import dataclass
from typing import Any, AnyStr, Callable, ForwardRef, NewType, TypeVar, get_type_hints
from docutils.frontend import OptionParser
from docutils.nodes import Node
from docutils.parsers.rst import Parser as RstParser
from docutils.utils import new_document
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import Options
from sphinx.ext.autodoc.mock import mock
from sphinx.util import logging
from sphinx.util.inspect import signature as sphinx_signature
from sphinx.util.inspect import stringify_signature
from .patches import install_patches
from .version import __version__
def backfill_type_hints(obj: Any, name: str) -> dict[str, Any]:
    parse_kwargs = {}
    if sys.version_info < (3, 8):
        try:
            import typed_ast.ast3 as ast
        except ImportError:
            return {}
    else:
        import ast
        parse_kwargs = {'type_comments': True}

    def _one_child(module: Module) -> stmt | None:
        children = module.body
        if len(children) != 1:
            _LOGGER.warning('Did not get exactly one node from AST for "%s", got %s', name, len(children))
            return None
        return children[0]
    try:
        code = textwrap.dedent(normalize_source_lines(inspect.getsource(obj)))
        obj_ast = ast.parse(code, **parse_kwargs)
    except (OSError, TypeError, SyntaxError):
        return {}
    obj_ast = _one_child(obj_ast)
    if obj_ast is None:
        return {}
    try:
        type_comment = obj_ast.type_comment
    except AttributeError:
        return {}
    if not type_comment:
        return {}
    try:
        comment_args_str, comment_returns = type_comment.split(' -> ')
    except ValueError:
        _LOGGER.warning('Unparseable type hint comment for "%s": Expected to contain ` -> `', name)
        return {}
    rv = {}
    if comment_returns:
        rv['return'] = comment_returns
    args = load_args(obj_ast)
    comment_args = split_type_comment_args(comment_args_str)
    is_inline = len(comment_args) == 1 and comment_args[0] == '...'
    if not is_inline:
        if args and args[0].arg in ('self', 'cls') and (len(comment_args) != len(args)):
            comment_args.insert(0, None)
        if len(args) != len(comment_args):
            _LOGGER.warning('Not enough type comments found on "%s"', name)
            return rv
    for at, arg in enumerate(args):
        arg_key = getattr(arg, 'arg', None)
        if arg_key is None:
            continue
        if is_inline:
            value = getattr(arg, 'type_comment', None)
        else:
            value = comment_args[at]
        if value is not None:
            rv[arg_key] = value
    return rv