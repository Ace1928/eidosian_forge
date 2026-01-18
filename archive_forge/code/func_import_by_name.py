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
def import_by_name(self, name: str, prefixes: List[Optional[str]]) -> Tuple[str, Any, Any, str]:
    with mock(self.config.autosummary_mock_imports):
        try:
            return import_by_name(name, prefixes)
        except ImportExceptionGroup as exc:
            try:
                return import_ivar_by_name(name, prefixes)
            except ImportError as exc2:
                if exc2.__cause__:
                    errors: List[BaseException] = exc.exceptions + [exc2.__cause__]
                else:
                    errors = exc.exceptions + [exc2]
                raise ImportExceptionGroup(exc.args[0], errors)