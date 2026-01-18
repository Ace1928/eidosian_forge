from __future__ import annotations
import datetime
import errno
import gettext
import hashlib
import hmac
import ipaddress
import json
import logging
import mimetypes
import os
import pathlib
import random
import re
import select
import signal
import socket
import stat
import sys
import threading
import time
import typing as t
import urllib
import warnings
from base64 import encodebytes
from pathlib import Path
import jupyter_client
from jupyter_client.kernelspec import KernelSpecManager
from jupyter_client.manager import KernelManager
from jupyter_client.session import Session
from jupyter_core.application import JupyterApp, base_aliases, base_flags
from jupyter_core.paths import jupyter_runtime_dir
from jupyter_events.logger import EventLogger
from nbformat.sign import NotebookNotary
from tornado import httpserver, ioloop, web
from tornado.httputil import url_concat
from tornado.log import LogFormatter, access_log, app_log, gen_log
from tornado.netutil import bind_sockets
from tornado.routing import Matcher, Rule
from traitlets import (
from traitlets.config import Config
from traitlets.config.application import boolean_flag, catch_config_error
from jupyter_server import (
from jupyter_server._sysinfo import get_sys_info
from jupyter_server._tz import utcnow
from jupyter_server.auth.authorizer import AllowAllAuthorizer, Authorizer
from jupyter_server.auth.identity import (
from jupyter_server.auth.login import LoginHandler
from jupyter_server.auth.logout import LogoutHandler
from jupyter_server.base.handlers import (
from jupyter_server.extension.config import ExtensionConfigManager
from jupyter_server.extension.manager import ExtensionManager
from jupyter_server.extension.serverextension import ServerExtensionApp
from jupyter_server.gateway.connections import GatewayWebSocketConnection
from jupyter_server.gateway.gateway_client import GatewayClient
from jupyter_server.gateway.managers import (
from jupyter_server.log import log_request
from jupyter_server.services.config import ConfigManager
from jupyter_server.services.contents.filemanager import (
from jupyter_server.services.contents.largefilemanager import AsyncLargeFileManager
from jupyter_server.services.contents.manager import AsyncContentsManager, ContentsManager
from jupyter_server.services.kernels.connection.base import BaseKernelWebsocketConnection
from jupyter_server.services.kernels.connection.channels import ZMQChannelsWebsocketConnection
from jupyter_server.services.kernels.kernelmanager import (
from jupyter_server.services.sessions.sessionmanager import SessionManager
from jupyter_server.utils import (
from jinja2 import Environment, FileSystemLoader
from jupyter_core.paths import secure_write
from jupyter_core.utils import ensure_async
from jupyter_server.transutils import _i18n, trans
from jupyter_server.utils import pathname2url, urljoin
def init_webapp(self) -> None:
    """initialize tornado webapp"""
    self.tornado_settings['allow_origin'] = self.allow_origin
    self.tornado_settings['websocket_compression_options'] = self.websocket_compression_options
    if self.allow_origin_pat:
        self.tornado_settings['allow_origin_pat'] = re.compile(self.allow_origin_pat)
    self.tornado_settings['allow_credentials'] = self.allow_credentials
    self.tornado_settings['autoreload'] = self.autoreload
    self.tornado_settings['cookie_options'] = self.identity_provider.cookie_options
    self.tornado_settings['get_secure_cookie_kwargs'] = self.identity_provider.get_secure_cookie_kwargs
    self.tornado_settings['token'] = self.identity_provider.token
    if self.static_immutable_cache:
        self.tornado_settings['static_immutable_cache'] = self.static_immutable_cache
    if not self.default_url.startswith(self.base_url):
        self.default_url = url_path_join(self.base_url, self.default_url)
    if self.sock:
        if self.port != DEFAULT_JUPYTER_SERVER_PORT:
            self.log.critical('Options --port and --sock are mutually exclusive. Aborting.')
            sys.exit(1)
        else:
            self.port = 0
        if self.open_browser:
            self.log.info('Ignoring --ServerApp.open_browser due to --sock being used.')
        if self.file_to_run:
            self.log.critical('Options --ServerApp.file_to_run and --sock are mutually exclusive.')
            sys.exit(1)
        if sys.platform.startswith('win'):
            self.log.critical('Option --sock is not supported on Windows, but got value of %s. Aborting.' % self.sock)
            sys.exit(1)
    self.web_app = ServerWebApplication(self, self.default_services, self.kernel_manager, self.contents_manager, self.session_manager, self.kernel_spec_manager, self.config_manager, self.event_logger, self.extra_services, self.log, self.base_url, self.default_url, self.tornado_settings, self.jinja_environment_options, authorizer=self.authorizer, identity_provider=self.identity_provider, kernel_websocket_connection_class=self.kernel_websocket_connection_class, websocket_ping_interval=self.websocket_ping_interval, websocket_ping_timeout=self.websocket_ping_timeout)
    if self.certfile:
        self.ssl_options['certfile'] = self.certfile
    if self.keyfile:
        self.ssl_options['keyfile'] = self.keyfile
    if self.client_ca:
        self.ssl_options['ca_certs'] = self.client_ca
    if not self.ssl_options:
        self.ssl_options = None
    else:
        import ssl
        self.ssl_options.setdefault('ssl_version', getattr(ssl, 'PROTOCOL_TLS', ssl.PROTOCOL_SSLv23))
        if self.ssl_options.get('ca_certs', False):
            self.ssl_options.setdefault('cert_reqs', ssl.CERT_REQUIRED)
    self.identity_provider.validate_security(self, ssl_options=self.ssl_options)
    if isinstance(self.identity_provider, LegacyIdentityProvider):
        self.identity_provider.settings = self.web_app.settings