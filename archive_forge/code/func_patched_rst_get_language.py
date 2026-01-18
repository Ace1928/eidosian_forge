import os
import re
import warnings
from contextlib import contextmanager
from copy import copy
from os import path
from types import ModuleType
from typing import (IO, TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Set,
import docutils
from docutils import nodes
from docutils.io import FileOutput
from docutils.nodes import Element, Node, system_message
from docutils.parsers.rst import Directive, directives, roles
from docutils.parsers.rst.states import Inliner
from docutils.statemachine import State, StateMachine, StringList
from docutils.utils import Reporter, unescape
from docutils.writers._html_base import HTMLTranslator
from sphinx.deprecation import RemovedInSphinx70Warning, deprecated_alias
from sphinx.errors import SphinxError
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.typing import RoleFunction
@contextmanager
def patched_rst_get_language() -> Generator[None, None, None]:
    """Patch docutils.parsers.rst.languages.get_language().
    Starting from docutils 0.17, get_language() in ``rst.languages``
    also has a reporter, which needs to be disabled temporarily.

    This should also work for old versions of docutils,
    because reporter is none by default.

    refs: https://github.com/sphinx-doc/sphinx/issues/10179
    """
    from docutils.parsers.rst.languages import get_language

    def patched_get_language(language_code: str, reporter: Reporter=None) -> Any:
        return get_language(language_code)
    try:
        docutils.parsers.rst.languages.get_language = patched_get_language
        yield
    finally:
        docutils.parsers.rst.languages.get_language = get_language