import functools
import os
import sys
import collections
import importlib
import warnings
from contextvars import copy_context
from importlib.machinery import ModuleSpec
import pkgutil
import threading
import re
import logging
import time
import mimetypes
import hashlib
import base64
import traceback
from urllib.parse import urlparse
from typing import Dict, Optional, Union
import flask
from importlib_metadata import version as _get_distribution_version
from dash import dcc
from dash import html
from dash import dash_table
from .fingerprint import build_fingerprint, check_fingerprint
from .resources import Scripts, Css
from .dependencies import (
from .development.base_component import ComponentRegistry
from .exceptions import (
from .version import __version__
from ._configs import get_combined_config, pathname_configs, pages_folder_config
from ._utils import (
from . import _callback
from . import _get_paths
from . import _dash_renderer
from . import _validate
from . import _watch
from . import _get_app
from ._grouping import map_grouping, grouping_len, update_args_group
from . import _pages
from ._pages import (
from ._jupyter import jupyter_dash, JupyterDisplayMode
from .types import RendererHooks
def serve_component_suites(self, package_name, fingerprinted_path):
    path_in_pkg, has_fingerprint = check_fingerprint(fingerprinted_path)
    _validate.validate_js_path(self.registered_paths, package_name, path_in_pkg)
    extension = '.' + path_in_pkg.split('.')[-1]
    mimetype = mimetypes.types_map.get(extension, 'application/octet-stream')
    package = sys.modules[package_name]
    self.logger.debug('serving -- package: %s[%s] resource: %s => location: %s', package_name, package.__version__, path_in_pkg, package.__path__)
    response = flask.Response(pkgutil.get_data(package_name, path_in_pkg), mimetype=mimetype)
    if has_fingerprint:
        response.cache_control.max_age = 31536000
    else:
        response.add_etag()
        tag = response.get_etag()[0]
        request_etag = flask.request.headers.get('If-None-Match')
        if f'"{tag}"' == request_etag:
            response = flask.Response(None, status=304)
    return response