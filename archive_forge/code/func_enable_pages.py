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
def enable_pages(self):
    if not self.use_pages:
        return
    if self.pages_folder:
        _import_layouts_from_pages(self.config.pages_folder)

    @self.server.before_request
    def router():
        if self._got_first_request['pages']:
            return
        self._got_first_request['pages'] = True
        inputs = {'pathname_': Input(_ID_LOCATION, 'pathname'), 'search_': Input(_ID_LOCATION, 'search')}
        inputs.update(self.routing_callback_inputs)

        @self.callback(Output(_ID_CONTENT, 'children'), Output(_ID_STORE, 'data'), inputs=inputs, prevent_initial_call=True)
        def update(pathname_, search_, **states):
            """
                Updates dash.page_container layout on page navigation.
                Updates the stored page title which will trigger the clientside callback to update the app title
                """
            query_parameters = _parse_query_string(search_)
            page, path_variables = _path_to_page(self.strip_relative_path(pathname_))
            if page == {}:
                for module, page in _pages.PAGE_REGISTRY.items():
                    if module.split('.')[-1] == 'not_found_404':
                        layout = page['layout']
                        title = page['title']
                        break
                else:
                    layout = html.H1('404 - Page not found')
                    title = self.title
            else:
                layout = page.get('layout', '')
                title = page['title']
            if callable(layout):
                layout = layout(**path_variables, **query_parameters, **states) if path_variables else layout(**query_parameters, **states)
            if callable(title):
                title = title(**path_variables) if path_variables else title()
            return (layout, {'title': title})
        _validate.check_for_duplicate_pathnames(_pages.PAGE_REGISTRY)
        _validate.validate_registry(_pages.PAGE_REGISTRY)
        if not self.config.suppress_callback_exceptions:
            self.validation_layout = html.Div([page['layout']() if callable(page['layout']) else page['layout'] for page in _pages.PAGE_REGISTRY.values()] + [self.layout() if callable(self.layout) else self.layout])
            if _ID_CONTENT not in self.validation_layout:
                raise Exception('`dash.page_container` not found in the layout')
        self.clientside_callback('\n                function(data) {{\n                    document.title = data.title\n                }}\n                ', Output(_ID_DUMMY, 'children'), Input(_ID_STORE, 'data'))