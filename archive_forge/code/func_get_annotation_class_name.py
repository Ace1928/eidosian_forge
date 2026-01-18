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
def get_annotation_class_name(annotation: Any, module: str) -> str:
    if annotation is None:
        return 'None'
    if annotation is AnyStr:
        return 'AnyStr'
    val = _get_types_type(annotation)
    if val is not None:
        return val
    if _is_newtype(annotation):
        return 'NewType'
    if getattr(annotation, '__qualname__', None):
        return annotation.__qualname__
    elif getattr(annotation, '_name', None):
        return annotation._name
    elif module in ('typing', 'typing_extensions') and isinstance(getattr(annotation, 'name', None), str):
        return annotation.name
    origin = getattr(annotation, '__origin__', None)
    if origin:
        if getattr(origin, '__qualname__', None):
            return origin.__qualname__
        elif getattr(origin, '_name', None):
            return origin._name
    annotation_cls = annotation if inspect.isclass(annotation) else type(annotation)
    return annotation_cls.__qualname__.lstrip('_')