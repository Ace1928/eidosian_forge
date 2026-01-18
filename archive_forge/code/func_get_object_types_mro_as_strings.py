from __future__ import absolute_import
import inspect
from inspect import cleandoc, getdoc, getfile, isclass, ismodule, signature
from typing import Any, Collection, Iterable, Optional, Tuple, Type, Union
from .console import Group, RenderableType
from .control import escape_control_codes
from .highlighter import ReprHighlighter
from .jupyter import JupyterMixin
from .panel import Panel
from .pretty import Pretty
from .table import Table
from .text import Text, TextType
def get_object_types_mro_as_strings(obj: object) -> Collection[str]:
    """
    Returns the MRO of an object's class as full qualified names, or of the object itself if it's a class.

    Examples:
        `object_types_mro_as_strings(JSONDecoder)` will return `['json.decoder.JSONDecoder', 'builtins.object']`
    """
    return [f'{getattr(type_, '__module__', '')}.{getattr(type_, '__qualname__', '')}' for type_ in get_object_types_mro(obj)]