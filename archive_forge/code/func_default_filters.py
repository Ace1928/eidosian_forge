from __future__ import annotations
import html
import json
import os
import typing as t
import uuid
import warnings
from pathlib import Path
from jinja2 import (
from jupyter_core.paths import jupyter_path
from nbformat import NotebookNode
from traitlets import Bool, Dict, HasTraits, List, Unicode, default, observe, validate
from traitlets.config import Config
from traitlets.utils.importstring import import_item
from nbconvert import filters
from .exporter import Exporter
def default_filters(self):
    """Override in subclasses to provide extra filters.

        This should return an iterable of 2-tuples: (name, class-or-function).
        You should call the method on the parent class and include the filters
        it provides.

        If a name is repeated, the last filter provided wins. Filters from
        user-supplied config win over filters provided by classes.
        """
    return default_filters.items()