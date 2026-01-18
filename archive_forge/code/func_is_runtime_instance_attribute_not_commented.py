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
def is_runtime_instance_attribute_not_commented(self, parent: Any) -> bool:
    """Check the subject is an attribute defined in __init__() without comment."""
    for cls in inspect.getmro(parent):
        try:
            module = safe_getattr(cls, '__module__')
            qualname = safe_getattr(cls, '__qualname__')
            analyzer = ModuleAnalyzer.for_module(module)
            analyzer.analyze()
            if qualname and self.objpath:
                key = '.'.join([qualname, self.objpath[-1]])
                if key in analyzer.tagorder:
                    return True
        except (AttributeError, PycodeError):
            pass
    return None