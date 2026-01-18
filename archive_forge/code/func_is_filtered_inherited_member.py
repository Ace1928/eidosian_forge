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
def is_filtered_inherited_member(name: str, obj: Any) -> bool:
    inherited_members = self.options.inherited_members or set()
    if inspect.isclass(self.object):
        for cls in self.object.__mro__:
            if cls.__name__ in inherited_members and cls != self.object:
                return True
            elif name in cls.__dict__:
                return False
            elif name in self.get_attr(cls, '__annotations__', {}):
                return False
            elif isinstance(obj, ObjectMember) and obj.class_ is cls:
                return False
    return False