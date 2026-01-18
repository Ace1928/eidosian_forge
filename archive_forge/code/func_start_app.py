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
def start_app(self) -> None:
    """Start the Jupyter Server application."""
    super().start()
    if not self.allow_root:
        try:
            uid = os.geteuid()
        except AttributeError:
            uid = -1
        if uid == 0:
            self.log.critical(_i18n('Running as root is not recommended. Use --allow-root to bypass.'))
            self.exit(1)
    info = self.log.info
    for line in self.running_server_info(kernel_count=False).split('\n'):
        info(line)
    info(_i18n('Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).'))
    if 'dev' in __version__:
        info(_i18n('Welcome to Project Jupyter! Explore the various tools available and their corresponding documentation. If you are interested in contributing to the platform, please visit the community resources section at https://jupyter.org/community.html.'))
    self.write_server_info_file()
    if not self.no_browser_open_file:
        self.write_browser_open_files()
    if self.open_browser and (not self.sock):
        self.launch_browser()
    if self.identity_provider.token and self.identity_provider.token_generated:
        if self.sock:
            self.log.critical('\n'.join(['\n', 'Jupyter Server is listening on %s' % self.display_url, '', f'UNIX sockets are not browser-connectable, but you can tunnel to the instance via e.g.`ssh -L 8888:{self.sock} -N user@this_host` and then open e.g. {self.connection_url} in a browser.']))
        else:
            if self.no_browser_open_file:
                message = ['\n', _i18n('To access the server, copy and paste one of these URLs:'), '    %s' % self.display_url]
            else:
                message = ['\n', _i18n('To access the server, open this file in a browser:'), '    %s' % urljoin('file:', pathname2url(self.browser_open_file)), _i18n('Or copy and paste one of these URLs:'), '    %s' % self.display_url]
            self.log.critical('\n'.join(message))