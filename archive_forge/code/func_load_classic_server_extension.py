from __future__ import annotations
import logging
import re
import sys
import typing as t
from jinja2 import Environment, FileSystemLoader
from jupyter_core.application import JupyterApp, NoStart
from tornado.log import LogFormatter
from tornado.web import RedirectHandler
from traitlets import Any, Bool, Dict, HasTraits, List, Unicode, default
from traitlets.config import Config
from jupyter_server.serverapp import ServerApp
from jupyter_server.transutils import _i18n
from jupyter_server.utils import is_namespace_package, url_path_join
from .handler import ExtensionHandlerMixin
@classmethod
def load_classic_server_extension(cls, serverapp):
    """Enables extension to be loaded as classic Notebook (jupyter/notebook) extension."""
    extension = cls()
    extension.serverapp = serverapp
    extension.load_config_file()
    extension.update_config(serverapp.config)
    extension.parse_command_line(serverapp.extra_args)
    extension.handlers.extend([('/static/favicons/favicon.ico', RedirectHandler, {'url': url_path_join(serverapp.base_url, 'static/base/images/favicon.ico')}), ('/static/favicons/favicon-busy-1.ico', RedirectHandler, {'url': url_path_join(serverapp.base_url, 'static/base/images/favicon-busy-1.ico')}), ('/static/favicons/favicon-busy-2.ico', RedirectHandler, {'url': url_path_join(serverapp.base_url, 'static/base/images/favicon-busy-2.ico')}), ('/static/favicons/favicon-busy-3.ico', RedirectHandler, {'url': url_path_join(serverapp.base_url, 'static/base/images/favicon-busy-3.ico')}), ('/static/favicons/favicon-file.ico', RedirectHandler, {'url': url_path_join(serverapp.base_url, 'static/base/images/favicon-file.ico')}), ('/static/favicons/favicon-notebook.ico', RedirectHandler, {'url': url_path_join(serverapp.base_url, 'static/base/images/favicon-notebook.ico')}), ('/static/favicons/favicon-terminal.ico', RedirectHandler, {'url': url_path_join(serverapp.base_url, 'static/base/images/favicon-terminal.ico')}), ('/static/logo/logo.png', RedirectHandler, {'url': url_path_join(serverapp.base_url, 'static/base/images/logo.png')})])
    extension.initialize()