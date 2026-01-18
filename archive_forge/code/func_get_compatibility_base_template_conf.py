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
@classmethod
def get_compatibility_base_template_conf(cls, name):
    """Get the base template config."""
    if name == 'display_priority':
        return {'base_template': 'base'}
    if name == 'full':
        return {'base_template': 'classic', 'mimetypes': {'text/html': True}}
    return None