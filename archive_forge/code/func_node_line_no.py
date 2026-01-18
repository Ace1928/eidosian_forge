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
def node_line_no(node: Node) -> int | None:
    """
    Get the 1-indexed line on which the node starts if possible. If not, return
    None.

    Descend through the first children until we locate one with a line number or
    return None if None of them have one.

    I'm not aware of any rst on which this returns None, to find out would
    require a more detailed analysis of the docutils rst parser source code. An
    example where the node doesn't have a line number but the first child does
    is all `definition_list` nodes. It seems like bullet_list and option_list
    get line numbers, but enum_list also doesn't. *shrug*.
    """
    while node.line is None and node.children:
        node = node.children[0]
    return node.line