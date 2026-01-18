import os
import pickle
import sys
import warnings
from collections import deque
from io import StringIO
from os import path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type, Union
from docutils import nodes
from docutils.nodes import Element, TextElement
from docutils.parsers import Parser
from docutils.parsers.rst import Directive, roles
from docutils.transforms import Transform
from pygments.lexer import Lexer
import sphinx
from sphinx import locale, package_dir
from sphinx.config import Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.domains import Domain, Index
from sphinx.environment import BuildEnvironment
from sphinx.environment.collectors import EnvironmentCollector
from sphinx.errors import ApplicationError, ConfigError, VersionRequirementError
from sphinx.events import EventManager
from sphinx.extension import Extension
from sphinx.highlighting import lexer_classes
from sphinx.locale import __
from sphinx.project import Project
from sphinx.registry import SphinxComponentRegistry
from sphinx.roles import XRefRole
from sphinx.theming import Theme
from sphinx.util import docutils, logging, progress_message
from sphinx.util.build_phase import BuildPhase
from sphinx.util.console import bold  # type: ignore
from sphinx.util.i18n import CatalogRepository
from sphinx.util.logging import prefixed_warnings
from sphinx.util.osutil import abspath, ensuredir, relpath
from sphinx.util.tags import Tags
from sphinx.util.typing import RoleFunction, TitleGetter
def set_translator(self, name: str, translator_class: Type[nodes.NodeVisitor], override: bool=False) -> None:
    """Register or override a Docutils translator class.

        This is used to register a custom output translator or to replace a
        builtin translator.  This allows extensions to use a custom translator
        and define custom nodes for the translator (see :meth:`add_node`).

        :param name: The name of the builder for the translator
        :param translator_class: A translator class
        :param override: If true, install the translator forcedly even if another translator
                         is already installed as the same name

        .. versionadded:: 1.3
        .. versionchanged:: 1.8
           Add *override* keyword.
        """
    self.registry.add_translator(name, translator_class, override=override)