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
def component_resource_path(component, attr, path):
    """
    Generates a canonical URL for a component resource.

    To be used in conjunction with the `panel.io.server.ComponentResourceHandler`
    which allows dynamically resolving resources defined on components.
    """
    if not isinstance(component, type):
        component = type(component)
    component_path = COMPONENT_PATH
    if state.rel_path:
        component_path = f'{state.rel_path}/{component_path}'
    rel_path = str(resolve_custom_path(component, path, relative=True)).replace(os.path.sep, '/')
    return f'{component_path}{component.__module__}/{component.__name__}/{attr}/{rel_path}'