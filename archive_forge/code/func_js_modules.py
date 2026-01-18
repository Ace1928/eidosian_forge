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
@property
def js_modules(self):
    from ..config import config
    from ..reactive import ReactiveHTML
    modules = list(config.js_modules.values())
    for model in Model.model_class_reverse_map.values():
        if hasattr(model, '__javascript_modules__'):
            modules.extend(model.__javascript_modules__)
    self.extra_resources(modules, '__javascript_modules__')
    if config.design:
        design_resources = config.design().resolve_resources(cdn=self.notebook or 'auto', include_theme=False)
        modules += [res for res in design_resources['js_modules'].values() if res not in modules]
    for model in param.concrete_descendents(ReactiveHTML).values():
        if not (getattr(model, '__javascript_modules__', None) and model._loaded()):
            continue
        for js_module in model.__javascript_modules__:
            if not isurl(js_module) and (not js_module.startswith('static/extensions')):
                js_module = component_resource_path(model, '__javascript_modules__', js_module)
            if js_module not in modules:
                modules.append(js_module)
    return self.adjust_paths(modules)