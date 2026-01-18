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
def find_autosummary_in_docstring(name: str, filename: Optional[str]=None) -> List[AutosummaryEntry]:
    """Find out what items are documented in the given object's docstring.

    See `find_autosummary_in_lines`.
    """
    try:
        real_name, obj, parent, modname = import_by_name(name)
        lines = pydoc.getdoc(obj).splitlines()
        return find_autosummary_in_lines(lines, module=name, filename=filename)
    except AttributeError:
        pass
    except ImportExceptionGroup as exc:
        errors = list({'* %s: %s' % (type(e).__name__, e) for e in exc.exceptions})
        print('Failed to import %s.\nPossible hints:\n%s' % (name, '\n'.join(errors)))
    except SystemExit:
        print("Failed to import '%s'; the module executes module level statement and it might call sys.exit()." % name)
    return []