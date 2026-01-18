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
def adjust_paths(self, resources):
    """
        Computes relative and absolute paths for resources.
        """
    new_resources = []
    version_suffix = f'?v={JS_VERSION}'
    cdn_base = f'{config.npm_cdn}/@holoviz/panel@{JS_VERSION}/dist/'
    for resource in resources:
        if not isinstance(resource, str):
            resource = str(resource)
        resource = resource.replace('https://unpkg.com', config.npm_cdn)
        if resource.startswith(cdn_base):
            resource = resource.replace(cdn_base, CDN_DIST)
        if self.mode == 'server':
            resource = resource.replace(CDN_DIST, LOCAL_DIST)
        if resource.startswith((state.base_url, 'static/')):
            if resource.startswith(state.base_url):
                resource = resource[len(state.base_url):]
            if state.rel_path:
                resource = f'{state.rel_path}/{resource}'
            elif self.absolute and self.mode == 'server':
                resource = f'{self.root_url}{resource}'
        if resource.endswith('.css'):
            resource += version_suffix
        new_resources.append(resource)
    return new_resources