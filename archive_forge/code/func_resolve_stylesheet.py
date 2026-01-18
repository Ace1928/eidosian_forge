from __future__ import annotations
import functools
import importlib
import json
import logging
import mimetypes
import os
import pathlib
import re
import textwrap
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import (
import param
from bokeh.embed.bundle import (
from bokeh.model import Model
from bokeh.models import ImportedStyleSheet
from bokeh.resources import Resources as BkResources, _get_server_urls
from bokeh.settings import settings as _settings
from jinja2.environment import Environment
from jinja2.loaders import FileSystemLoader
from markupsafe import Markup
from ..config import config, panel_extension as extension
from ..util import isurl, url_path
from .state import state
def resolve_stylesheet(cls, stylesheet: str, attribute: str | None=None):
    """
    Resolves a stylesheet definition, e.g. originating on a component
    Reactive._stylesheets or a Design.modifiers attribute. Stylesheets
    may be defined as one of the following:

    - Absolute URL defined with http(s) protocol
    - A path relative to the component
    - A raw css string

    Arguments
    ---------
    cls: type | object
        Object or class defining the stylesheet
    stylesheet: str
        The stylesheet definition
    """
    stylesheet = str(stylesheet)
    if not stylesheet.startswith('http') and attribute and _is_file_path(stylesheet) and (custom_path := resolve_custom_path(cls, stylesheet)):
        if not state._is_pyodide and state.curdoc and state.curdoc.session_context:
            stylesheet = component_resource_path(cls, attribute, stylesheet)
        else:
            stylesheet = custom_path.read_text(encoding='utf-8')
    return stylesheet