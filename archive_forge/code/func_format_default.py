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
def format_default(app: Sphinx, default: Any, is_annotated: bool) -> str | None:
    if default is inspect.Parameter.empty:
        return None
    formatted = repr(default).replace('\\', '\\\\')
    if is_annotated:
        if app.config.typehints_defaults.startswith('braces'):
            return f' (default: ``{formatted}``)'
        else:
            return f', default: ``{formatted}``'
    elif app.config.typehints_defaults == 'braces-after':
        return f' (default: ``{formatted}``)'
    else:
        return f'default: ``{formatted}``'