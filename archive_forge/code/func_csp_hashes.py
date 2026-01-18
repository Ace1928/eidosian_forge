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
def csp_hashes(self, hash_algorithm='sha256'):
    """Calculates CSP hashes (sha + base64) of all inline scripts, such that
        one of the biggest benefits of CSP (disallowing general inline scripts)
        can be utilized together with Dash clientside callbacks (inline scripts).

        Calculate these hashes after all inline callbacks are defined,
        and add them to your CSP headers before starting the server, for example
        with the flask-talisman package from PyPI:

        flask_talisman.Talisman(app.server, content_security_policy={
            "default-src": "'self'",
            "script-src": ["'self'"] + app.csp_hashes()
        })

        :param hash_algorithm: One of the recognized CSP hash algorithms ('sha256', 'sha384', 'sha512').
        :return: List of CSP hash strings of all inline scripts.
        """
    HASH_ALGORITHMS = ['sha256', 'sha384', 'sha512']
    if hash_algorithm not in HASH_ALGORITHMS:
        raise ValueError('Possible CSP hash algorithms: ' + ', '.join(HASH_ALGORITHMS))
    method = getattr(hashlib, hash_algorithm)

    def _hash(script):
        return base64.b64encode(method(script.encode('utf-8')).digest()).decode('utf-8')
    self._inline_scripts.extend(_callback.GLOBAL_INLINE_SCRIPTS)
    _callback.GLOBAL_INLINE_SCRIPTS.clear()
    return [f"'{hash_algorithm}-{_hash(script)}'" for script in self._inline_scripts + [self.renderer]]