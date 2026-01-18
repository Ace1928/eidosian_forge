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
def import_object(self, raiseerror: bool=False) -> bool:
    """Check the exisitence of uninitialized instance attribute when failed to import
        the attribute."""
    ret = super().import_object(raiseerror)
    if ret and (not inspect.isproperty(self.object)):
        __dict__ = safe_getattr(self.parent, '__dict__', {})
        obj = __dict__.get(self.objpath[-1])
        if isinstance(obj, classmethod) and inspect.isproperty(obj.__func__):
            self.object = obj.__func__
            self.isclassmethod = True
            return True
        else:
            return False
    self.isclassmethod = False
    return ret