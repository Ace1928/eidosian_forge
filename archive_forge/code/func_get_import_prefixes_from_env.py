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
def get_import_prefixes_from_env(env: BuildEnvironment) -> List[Optional[str]]:
    """
    Obtain current Python import prefixes (for `import_by_name`)
    from ``document.env``
    """
    prefixes: List[Optional[str]] = [None]
    currmodule = env.ref_context.get('py:module')
    if currmodule:
        prefixes.insert(0, currmodule)
    currclass = env.ref_context.get('py:class')
    if currclass:
        if currmodule:
            prefixes.insert(0, currmodule + '.' + currclass)
        else:
            prefixes.insert(0, currclass)
    return prefixes