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
def annotate_to_first_argument(self, func: Callable, typ: Type) -> Optional[Callable]:
    """Annotate type hint to the first argument of function if needed."""
    try:
        sig = inspect.signature(func, type_aliases=self.config.autodoc_type_aliases)
    except TypeError as exc:
        logger.warning(__('Failed to get a method signature for %s: %s'), self.fullname, exc)
        return None
    except ValueError:
        return None
    if len(sig.parameters) == 1:
        return None

    def dummy():
        pass
    params = list(sig.parameters.values())
    if params[1].annotation is Parameter.empty:
        params[1] = params[1].replace(annotation=typ)
        try:
            dummy.__signature__ = sig.replace(parameters=params)
            return dummy
        except (AttributeError, TypeError):
            return None
    return func