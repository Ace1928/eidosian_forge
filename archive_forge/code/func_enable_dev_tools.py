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
def enable_dev_tools(self, debug=None, dev_tools_ui=None, dev_tools_props_check=None, dev_tools_serve_dev_bundles=None, dev_tools_hot_reload=None, dev_tools_hot_reload_interval=None, dev_tools_hot_reload_watch_interval=None, dev_tools_hot_reload_max_retry=None, dev_tools_silence_routes_logging=None, dev_tools_prune_errors=None):
    """Activate the dev tools, called by `run`. If your application
        is served by wsgi and you want to activate the dev tools, you can call
        this method out of `__main__`.

        All parameters can be set by environment variables as listed.
        Values provided here take precedence over environment variables.

        Available dev_tools environment variables:

            - DASH_DEBUG
            - DASH_UI
            - DASH_PROPS_CHECK
            - DASH_SERVE_DEV_BUNDLES
            - DASH_HOT_RELOAD
            - DASH_HOT_RELOAD_INTERVAL
            - DASH_HOT_RELOAD_WATCH_INTERVAL
            - DASH_HOT_RELOAD_MAX_RETRY
            - DASH_SILENCE_ROUTES_LOGGING
            - DASH_PRUNE_ERRORS

        :param debug: Enable/disable all the dev tools unless overridden by the
            arguments or environment variables. Default is ``True`` when
            ``enable_dev_tools`` is called directly, and ``False`` when called
            via ``run``. env: ``DASH_DEBUG``
        :type debug: bool

        :param dev_tools_ui: Show the dev tools UI. env: ``DASH_UI``
        :type dev_tools_ui: bool

        :param dev_tools_props_check: Validate the types and values of Dash
            component props. env: ``DASH_PROPS_CHECK``
        :type dev_tools_props_check: bool

        :param dev_tools_serve_dev_bundles: Serve the dev bundles. Production
            bundles do not necessarily include all the dev tools code.
            env: ``DASH_SERVE_DEV_BUNDLES``
        :type dev_tools_serve_dev_bundles: bool

        :param dev_tools_hot_reload: Activate hot reloading when app, assets,
            and component files change. env: ``DASH_HOT_RELOAD``
        :type dev_tools_hot_reload: bool

        :param dev_tools_hot_reload_interval: Interval in seconds for the
            client to request the reload hash. Default 3.
            env: ``DASH_HOT_RELOAD_INTERVAL``
        :type dev_tools_hot_reload_interval: float

        :param dev_tools_hot_reload_watch_interval: Interval in seconds for the
            server to check asset and component folders for changes.
            Default 0.5. env: ``DASH_HOT_RELOAD_WATCH_INTERVAL``
        :type dev_tools_hot_reload_watch_interval: float

        :param dev_tools_hot_reload_max_retry: Maximum number of failed reload
            hash requests before failing and displaying a pop up. Default 8.
            env: ``DASH_HOT_RELOAD_MAX_RETRY``
        :type dev_tools_hot_reload_max_retry: int

        :param dev_tools_silence_routes_logging: Silence the `werkzeug` logger,
            will remove all routes logging. Enabled with debugging by default
            because hot reload hash checks generate a lot of requests.
            env: ``DASH_SILENCE_ROUTES_LOGGING``
        :type dev_tools_silence_routes_logging: bool

        :param dev_tools_prune_errors: Reduce tracebacks to just user code,
            stripping out Flask and Dash pieces. Only available with debugging.
            `True` by default, set to `False` to see the complete traceback.
            env: ``DASH_PRUNE_ERRORS``
        :type dev_tools_prune_errors: bool

        :return: debug
        """
    if debug is None:
        debug = get_combined_config('debug', None, True)
    dev_tools = self._setup_dev_tools(debug=debug, ui=dev_tools_ui, props_check=dev_tools_props_check, serve_dev_bundles=dev_tools_serve_dev_bundles, hot_reload=dev_tools_hot_reload, hot_reload_interval=dev_tools_hot_reload_interval, hot_reload_watch_interval=dev_tools_hot_reload_watch_interval, hot_reload_max_retry=dev_tools_hot_reload_max_retry, silence_routes_logging=dev_tools_silence_routes_logging, prune_errors=dev_tools_prune_errors)
    if dev_tools.silence_routes_logging:
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
    if dev_tools.hot_reload:
        _reload = self._hot_reload
        _reload.hash = generate_hash()
        packages = [pkgutil.find_loader(x) for x in list(ComponentRegistry.registry) if x != '__main__']
        if '_pytest' in sys.modules:
            from _pytest.assertion.rewrite import AssertionRewritingHook
            for index, package in enumerate(packages):
                if isinstance(package, AssertionRewritingHook):
                    dash_spec = importlib.util.find_spec('dash')
                    dash_test_path = dash_spec.submodule_search_locations[0]
                    setattr(dash_spec, 'path', dash_test_path)
                    packages[index] = dash_spec
        component_packages_dist = [dash_test_path if isinstance(package, ModuleSpec) else os.path.dirname(package.path) if hasattr(package, 'path') else os.path.dirname(package._path[0]) if hasattr(package, '_path') else package.filename for package in packages]
        for i, package in enumerate(packages):
            if hasattr(package, 'path') and 'dash/dash' in os.path.dirname(package.path):
                component_packages_dist[i:i + 1] = [os.path.join(os.path.dirname(package.path), x) for x in ['dcc', 'html', 'dash_table']]
        _reload.watch_thread = threading.Thread(target=lambda: _watch.watch([self.config.assets_folder] + component_packages_dist, self._on_assets_change, sleep_time=dev_tools.hot_reload_watch_interval))
        _reload.watch_thread.daemon = True
        _reload.watch_thread.start()
    if debug:
        if jupyter_dash.active:
            jupyter_dash.configure_callback_exception_handling(self, dev_tools.prune_errors)
        elif dev_tools.prune_errors:
            secret = gen_salt(20)

            @self.server.errorhandler(Exception)
            def _wrap_errors(error):
                tb = _get_traceback(secret, error)
                return (tb, 500)
    if debug and dev_tools.ui:

        def _before_request():
            flask.g.timing_information = {'__dash_server': {'dur': time.time(), 'desc': None}}

        def _after_request(response):
            timing_information = flask.g.get('timing_information', None)
            if timing_information is None:
                return response
            dash_total = timing_information.get('__dash_server', None)
            if dash_total is not None:
                dash_total['dur'] = round((time.time() - dash_total['dur']) * 1000)
            for name, info in timing_information.items():
                value = name
                if info.get('desc') is not None:
                    value += f';desc="{info['desc']}"'
                if info.get('dur') is not None:
                    value += f';dur={info['dur']}'
                response.headers.add('Server-Timing', value)
            return response
        self.server.before_request(_before_request)
        self.server.after_request(_after_request)
    if debug and dev_tools.serve_dev_bundles and (not self.scripts.config.serve_locally):
        self.scripts.config.serve_locally = True
        print('WARNING: dev bundles requested with serve_locally=False.\nThis is not supported, switching to serve_locally=True')
    return debug