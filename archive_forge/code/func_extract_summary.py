import inspect
import os
import posixpath
import re
import sys
import warnings
from inspect import Parameter
from os import path
from types import ModuleType
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, cast
from docutils import nodes
from docutils.nodes import Node, system_message
from docutils.parsers.rst import directives
from docutils.parsers.rst.states import RSTStateMachine, Struct, state_classes
from docutils.statemachine import StringList
import sphinx
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.config import Config
from sphinx.deprecation import (RemovedInSphinx60Warning, RemovedInSphinx70Warning,
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc import INSTANCEATTR, Documenter
from sphinx.ext.autodoc.directive import DocumenterBridge, Options
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autodoc.mock import mock
from sphinx.extension import Extension
from sphinx.locale import __
from sphinx.project import Project
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst
from sphinx.util.docutils import (NullReporter, SphinxDirective, SphinxRole, new_document,
from sphinx.util.inspect import signature_from_str
from sphinx.util.matching import Matcher
from sphinx.util.typing import OptionSpec
from sphinx.writers.html import HTMLTranslator
def extract_summary(doc: List[str], document: Any) -> str:
    """Extract summary from docstring."""

    def parse(doc: List[str], settings: Any) -> nodes.document:
        state_machine = RSTStateMachine(state_classes, 'Body')
        node = new_document('', settings)
        node.reporter = NullReporter()
        state_machine.run(doc, node)
        return node
    while doc and (not doc[0].strip()):
        doc.pop(0)
    for i, piece in enumerate(doc):
        if not piece.strip():
            doc = doc[:i]
            break
    if doc == []:
        return ''
    node = parse(doc, document.settings)
    if isinstance(node[0], nodes.section):
        summary = node[0].astext().strip()
    elif not isinstance(node[0], nodes.paragraph):
        summary = doc[0].strip()
    else:
        sentences = periods_re.split(' '.join(doc))
        if len(sentences) == 1:
            summary = sentences[0].strip()
        else:
            summary = ''
            for i in range(len(sentences)):
                summary = '. '.join(sentences[:i + 1]).rstrip('.') + '.'
                node[:] = []
                node = parse(doc, document.settings)
                if summary.endswith(WELL_KNOWN_ABBREVIATIONS):
                    pass
                elif not any(node.findall(nodes.system_message)):
                    break
    summary = literal_re.sub('.', summary)
    return summary