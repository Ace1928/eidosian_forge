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
def bundled_files(model, file_type='javascript'):
    name = model.__name__.lower()
    bdir = BUNDLE_DIR / name
    shared = list((JS_URLS if file_type == 'javascript' else CSS_URLS).values())
    files = []
    for url in getattr(model, f'__{file_type}_raw__', []):
        if url.startswith(CDN_DIST):
            filepath = url.replace(f'{CDN_DIST}bundled/', '')
        elif url.startswith(config.npm_cdn):
            filepath = url.replace(config.npm_cdn, '')[1:]
        else:
            filepath = url_path(url)
        test_filepath = filepath.split('?')[0]
        if url in shared:
            prefixed = filepath
            test_path = BUNDLE_DIR / test_filepath
        elif not test_filepath.replace('/', '').startswith(f'{name}/'):
            prefixed = f'{name}/{test_filepath}'
            test_path = bdir / test_filepath
        else:
            prefixed = test_filepath
            test_path = BUNDLE_DIR / test_filepath
        if test_path.is_file():
            if RESOURCE_MODE == 'server':
                files.append(f'static/extensions/panel/bundled/{prefixed}')
            elif filepath == test_filepath:
                files.append(f'{CDN_DIST}bundled/{prefixed}')
            else:
                files.append(url)
        else:
            files.append(url)
    return files