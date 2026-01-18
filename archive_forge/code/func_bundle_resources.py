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
def bundle_resources(roots, resources, notebook=False, reloading=False, enable_mathjax='auto'):
    from ..config import panel_extension as ext
    global RESOURCE_MODE
    if not isinstance(resources, Resources):
        resources = Resources.from_bokeh(resources, notebook=notebook)
    js_resources = css_resources = resources
    RESOURCE_MODE = mode = js_resources.mode if resources is not None else 'inline'
    js_files = []
    js_raw = []
    css_files = []
    css_raw = []
    if isinstance(enable_mathjax, bool):
        use_mathjax = enable_mathjax
    elif roots:
        use_mathjax = _use_mathjax(roots) or 'mathjax' in ext._loaded_extensions
    else:
        use_mathjax = 'mathjax' in ext._loaded_extensions
    if js_resources:
        js_resources = js_resources.clone()
        if not use_mathjax and 'bokeh-mathjax' in js_resources.components:
            js_resources.components.remove('bokeh-mathjax')
        if reloading:
            js_resources.components.clear()
        js_files.extend(js_resources.js_files)
        js_raw.extend(js_resources.js_raw)
    css_files.extend(css_resources.css_files)
    css_raw.extend(css_resources.css_raw)
    extensions = _bundle_extensions(None, js_resources)
    if reloading:
        extensions = [ext for ext in extensions if not (ext.cdn_url is not None and str(ext.cdn_url).startswith('https://unpkg.com/@holoviz/panel@'))]
    extra_js = []
    if mode == 'inline':
        js_raw.extend([Resources._inline(bundle.artifact_path) for bundle in extensions])
    elif mode == 'server':
        for bundle in extensions:
            server_url = bundle.server_url
            if not isinstance(server_url, str):
                server_url = str(server_url)
            if resources.root_url and (not resources.absolute):
                server_url = server_url.replace(resources.root_url, '', 1)
                if state.rel_path:
                    server_url = f'{state.rel_path}/{server_url}'
            js_files.append(server_url)
    elif mode == 'cdn':
        for bundle in extensions:
            if bundle.cdn_url is not None:
                extra_js.append(bundle.cdn_url)
            else:
                js_raw.append(Resources._inline(bundle.artifact_path))
    else:
        extra_js.extend([bundle.artifact_path for bundle in extensions])
    js_files += resources.adjust_paths(extra_js)
    ext = bundle_models(None)
    if ext is not None:
        js_raw.append(ext)
    hashes = js_resources.hashes if js_resources else {}
    js_files = list(map(URL, js_files))
    css_files = list(map(URL, css_files))
    return Bundle(css_files=css_files, css_raw=css_raw, hashes=hashes, js_files=js_files, js_raw=js_raw, js_module_exports=resources.js_module_exports, js_modules=resources.js_modules, notebook=notebook)