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
@classmethod
def from_bokeh(cls, bk_bundle, notebook=False):
    return cls(notebook=notebook, js_files=bk_bundle.js_files, js_raw=bk_bundle.js_raw, css_files=bk_bundle.css_files, css_raw=bk_bundle.css_raw, hashes=bk_bundle.hashes)