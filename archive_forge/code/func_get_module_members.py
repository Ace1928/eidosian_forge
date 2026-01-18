import argparse
import inspect
import locale
import os
import pkgutil
import pydoc
import re
import sys
from gettext import NullTranslations
from os import path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Type
from jinja2 import TemplateNotFound
from jinja2.sandbox import SandboxedEnvironment
import sphinx.locale
from sphinx import __display_version__, package_dir
from sphinx.application import Sphinx
from sphinx.builders import Builder
from sphinx.config import Config
from sphinx.ext.autodoc import Documenter
from sphinx.ext.autodoc.importer import import_module
from sphinx.ext.autosummary import (ImportExceptionGroup, get_documenter, import_by_name,
from sphinx.locale import __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.registry import SphinxComponentRegistry
from sphinx.util import logging, rst, split_full_qualified_name
from sphinx.util.inspect import getall, safe_getattr
from sphinx.util.osutil import ensuredir
from sphinx.util.template import SphinxTemplateLoader
def get_module_members(obj: Any) -> Dict[str, Any]:
    members = {}
    for name in members_of(obj, app.config):
        try:
            members[name] = safe_getattr(obj, name)
        except AttributeError:
            continue
    return members