import collections
import inspect
import re
import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Tuple, Type, Union
from sphinx.application import Sphinx
from sphinx.config import Config as SphinxConfig
from sphinx.deprecation import RemovedInSphinx60Warning
from sphinx.locale import _, __
from sphinx.util import logging
from sphinx.util.inspect import stringify_annotation
from sphinx.util.typing import get_type_hints
def push_item(name: str, rest: List[str]) -> None:
    if not name:
        return
    name, role = parse_item_name(name)
    items.append((name, list(rest), role))
    del rest[:]