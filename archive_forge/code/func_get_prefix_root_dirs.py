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
def get_prefix_root_dirs(self):
    """Get the prefix root dirs."""
    root_dirs = []
    if DEV_MODE:
        root_dirs.append(os.path.abspath(os.path.join(ROOT, '..', '..', 'share', 'jupyter')))
    root_dirs.extend(jupyter_path())
    return root_dirs