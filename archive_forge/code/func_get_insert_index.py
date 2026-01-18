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
def get_insert_index(app: Sphinx, lines: list[str]) -> InsertIndexInfo | None:
    if any((line.startswith(':rtype:') for line in lines)):
        return None
    for at, line in enumerate(lines):
        if line.startswith((':return:', ':returns:')):
            return InsertIndexInfo(insert_index=at, found_return=True)
    settings = OptionParser(components=(RstParser,)).get_default_values()
    settings.env = app.env
    doc = new_document('', settings=settings)
    RstParser().parse('\n'.join(lines), doc)
    for child in doc.children:
        if tag_name(child) != 'field_list':
            continue
        if not any((c.children[0].astext().startswith(PARAM_SYNONYMS) for c in child.children)):
            continue
        next_sibling = child.next_node(descend=False, siblings=True)
        line_no = node_line_no(next_sibling) if next_sibling else None
        at = line_no - 2 if line_no else len(lines)
        return InsertIndexInfo(insert_index=at, found_param=True)
    for child in doc.children:
        if tag_name(child) in ['literal_block', 'paragraph', 'field_list']:
            continue
        line_no = node_line_no(child)
        at = line_no - 2 if line_no else len(lines)
        return InsertIndexInfo(insert_index=at, found_directive=True)
    return InsertIndexInfo(insert_index=len(lines))