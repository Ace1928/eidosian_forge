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
def js_files(self):
    from ..config import config
    files = super(Resources, self).js_files
    self.extra_resources(files, '__javascript__')
    files += [js for js in config.js_files.values()]
    if config.design:
        design_js = config.design().resolve_resources(cdn=self.notebook or 'auto', include_theme=False)['js'].values()
        files += [res for res in design_js if res not in files]
    js_files = self.adjust_paths([js for js in files if self.mode != 'inline' or not is_cdn_url(js)])
    dist_dir = self.dist_dir
    require_index = [i for i, jsf in enumerate(js_files) if 'require' in jsf]
    if require_index:
        requirejs = js_files.pop(require_index[0])
        if any(('ace' in jsf for jsf in js_files)):
            js_files.append(dist_dir + 'pre_require.js')
        js_files.append(requirejs)
        if any(('ace' in jsf for jsf in js_files)):
            js_files.append(dist_dir + 'post_require.js')
    return js_files