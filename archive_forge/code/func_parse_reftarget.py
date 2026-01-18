import builtins
import inspect
import re
import sys
import typing
import warnings
from inspect import Parameter
from typing import Any, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Element, Node
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import Inliner
from sphinx import addnodes
from sphinx.addnodes import desc_signature, pending_xref, pending_xref_condition
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.directives import ObjectDescription
from sphinx.domains import Domain, Index, IndexEntry, ObjType
from sphinx.environment import BuildEnvironment
from sphinx.locale import _, __
from sphinx.pycode.ast import ast
from sphinx.pycode.ast import parse as ast_parse
from sphinx.roles import XRefRole
from sphinx.util import logging
from sphinx.util.docfields import Field, GroupedField, TypedField
from sphinx.util.docutils import SphinxDirective, switch_source_input
from sphinx.util.inspect import signature_from_str
from sphinx.util.nodes import (find_pending_xref_condition, make_id, make_refnode,
from sphinx.util.typing import OptionSpec, TextlikeNode
def parse_reftarget(reftarget: str, suppress_prefix: bool=False) -> Tuple[str, str, str, bool]:
    """Parse a type string and return (reftype, reftarget, title, refspecific flag)"""
    refspecific = False
    if reftarget.startswith('.'):
        reftarget = reftarget[1:]
        title = reftarget
        refspecific = True
    elif reftarget.startswith('~'):
        reftarget = reftarget[1:]
        title = reftarget.split('.')[-1]
    elif suppress_prefix:
        title = reftarget.split('.')[-1]
    elif reftarget.startswith('typing.'):
        title = reftarget[7:]
    else:
        title = reftarget
    if reftarget == 'None' or reftarget.startswith('typing.'):
        reftype = 'obj'
    else:
        reftype = 'class'
    return (reftype, reftarget, title, refspecific)