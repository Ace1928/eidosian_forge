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
def format_annotation(annotation: Any, config: Config) -> str:
    typehints_formatter: Callable[..., str] | None = getattr(config, 'typehints_formatter', None)
    if typehints_formatter is not None:
        formatted = typehints_formatter(annotation, config)
        if formatted is not None:
            return formatted
    if isinstance(annotation, ForwardRef):
        return annotation.__forward_arg__
    if annotation is None or annotation is type(None):
        return ':py:obj:`None`'
    if annotation is Ellipsis:
        return ':py:data:`...<Ellipsis>`'
    if isinstance(annotation, tuple):
        return format_internal_tuple(annotation, config)
    try:
        module = get_annotation_module(annotation)
        class_name = get_annotation_class_name(annotation, module)
        args = get_annotation_args(annotation, module, class_name)
    except ValueError:
        return str(annotation).strip("'")
    if module == 'typing_extensions':
        module = 'typing'
    if module == '_io':
        module = 'io'
    full_name = f'{module}.{class_name}' if module != 'builtins' else class_name
    fully_qualified: bool = getattr(config, 'typehints_fully_qualified', False)
    prefix = '' if fully_qualified or full_name == class_name else '~'
    if module == 'typing' and class_name in _PYDATA_ANNOTATIONS:
        role = 'data'
    else:
        role = 'class'
    args_format = '\\[{}]'
    formatted_args: str | None = ''
    if full_name == 'typing.NewType':
        args_format = f'\\(``{annotation.__name__}``, {{}})'
        role = 'class' if sys.version_info >= (3, 10) else 'func'
    elif full_name in {'typing.TypeVar', 'typing.ParamSpec'}:
        params = {k: getattr(annotation, f'__{k}__') for k in ('bound', 'covariant', 'contravariant')}
        params = {k: v for k, v in params.items() if v}
        if 'bound' in params:
            params['bound'] = f' {format_annotation(params['bound'], config)}'
        args_format = f'\\(``{annotation.__name__}``{(', {}' if args else '')}'
        if params:
            args_format += ''.join((f', {k}={v}' for k, v in params.items()))
        args_format += ')'
        formatted_args = None if args else args_format
    elif full_name == 'typing.Optional':
        args = tuple((x for x in args if x is not type(None)))
    elif full_name in ('typing.Union', 'types.UnionType') and type(None) in args:
        if len(args) == 2:
            full_name = 'typing.Optional'
            role = 'data'
            args = tuple((x for x in args if x is not type(None)))
        else:
            simplify_optional_unions: bool = getattr(config, 'simplify_optional_unions', True)
            if not simplify_optional_unions:
                full_name = 'typing.Optional'
                role = 'data'
                args_format = f'\\[:py:data:`{prefix}typing.Union`\\[{{}}]]'
                args = tuple((x for x in args if x is not type(None)))
    elif full_name in ('typing.Callable', 'collections.abc.Callable') and args and (args[0] is not ...):
        fmt = [format_annotation(arg, config) for arg in args]
        formatted_args = f'\\[\\[{', '.join(fmt[:-1])}], {fmt[-1]}]'
    elif full_name == 'typing.Literal':
        formatted_args = '\\[{}]'.format(', '.join((f'``{arg!r}``' for arg in args)))
    elif full_name == 'types.UnionType':
        return ' | '.join([format_annotation(arg, config) for arg in args])
    if args and (not formatted_args):
        try:
            iter(args)
        except TypeError:
            fmt = [format_annotation(args, config)]
        else:
            fmt = [format_annotation(arg, config) for arg in args]
        formatted_args = args_format.format(', '.join(fmt))
    result = f':py:{role}:`{prefix}{full_name}`{formatted_args}'
    return result