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
def process_generate_options(app: Sphinx) -> None:
    genfiles = app.config.autosummary_generate
    if genfiles is True:
        env = app.builder.env
        genfiles = [env.doc2path(x, base=False) for x in env.found_docs if os.path.isfile(env.doc2path(x))]
    elif genfiles is False:
        pass
    else:
        ext = list(app.config.source_suffix)
        genfiles = [genfile + (ext[0] if not genfile.endswith(tuple(ext)) else '') for genfile in genfiles]
        for entry in genfiles[:]:
            if not path.isfile(path.join(app.srcdir, entry)):
                logger.warning(__('autosummary_generate: file not found: %s'), entry)
                genfiles.remove(entry)
    if not genfiles:
        return
    suffix = get_rst_suffix(app)
    if suffix is None:
        logger.warning(__('autosummary generats .rst files internally. But your source_suffix does not contain .rst. Skipped.'))
        return
    from sphinx.ext.autosummary.generate import generate_autosummary_docs
    imported_members = app.config.autosummary_imported_members
    with mock(app.config.autosummary_mock_imports):
        generate_autosummary_docs(genfiles, suffix=suffix, base_path=app.srcdir, app=app, imported_members=imported_members, overwrite=app.config.autosummary_generate_overwrite, encoding=app.config.source_encoding)