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
def import_ivar_by_name(name: str, prefixes: List[Optional[str]]=[None], grouped_exception: bool=True) -> Tuple[str, Any, Any, str]:
    """Import an instance variable that has the given *name*, under one of the
    *prefixes*.  The first name that succeeds is used.
    """
    try:
        name, attr = name.rsplit('.', 1)
        real_name, obj, parent, modname = import_by_name(name, prefixes, grouped_exception)
        qualname = real_name.replace(modname + '.', '')
        analyzer = ModuleAnalyzer.for_module(getattr(obj, '__module__', modname))
        analyzer.analyze()
        if (qualname, attr) in analyzer.attr_docs or (qualname, attr) in analyzer.annotations:
            return (real_name + '.' + attr, INSTANCEATTR, obj, modname)
    except (ImportError, ValueError, PycodeError) as exc:
        raise ImportError from exc
    except ImportExceptionGroup:
        raise
    raise ImportError