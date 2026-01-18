import re
import warnings
from inspect import Parameter, Signature
from types import ModuleType
from typing import (TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence,
from docutils.statemachine import StringList
import sphinx
from sphinx.application import Sphinx
from sphinx.config import ENUM, Config
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.environment import BuildEnvironment
from sphinx.ext.autodoc.importer import (get_class_members, get_object_members, import_module,
from sphinx.ext.autodoc.mock import ismock, mock, undecorate
from sphinx.locale import _, __
from sphinx.pycode import ModuleAnalyzer, PycodeError
from sphinx.util import inspect, logging
from sphinx.util.docstrings import prepare_docstring, separate_metadata
from sphinx.util.inspect import (evaluate_signature, getdoc, object_description, safe_getattr,
from sphinx.util.typing import OptionSpec, get_type_hints, restify
from sphinx.util.typing import stringify as stringify_typehint
def get_canonical_fullname(self) -> Optional[str]:
    __modname__ = safe_getattr(self.object, '__module__', self.modname)
    __qualname__ = safe_getattr(self.object, '__qualname__', None)
    if __qualname__ is None:
        __qualname__ = safe_getattr(self.object, '__name__', None)
    if __qualname__ and '<locals>' in __qualname__:
        __qualname__ = None
    if __modname__ and __qualname__:
        return '.'.join([__modname__, __qualname__])
    else:
        return None