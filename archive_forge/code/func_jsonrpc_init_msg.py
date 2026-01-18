import json
import os
import pathlib
import shutil
from pathlib import Path
from typing import Text
from jupyter_server.serverapp import ServerApp
from pytest import fixture
from tornado.httpserver import HTTPRequest
from tornado.httputil import HTTPServerRequest
from tornado.queues import Queue
from tornado.web import Application
from jupyter_lsp import LanguageServerManager
from jupyter_lsp.constants import APP_CONFIG_D_SECTIONS
from jupyter_lsp.handlers import LanguageServersHandler, LanguageServerWebSocketHandler
@fixture
def jsonrpc_init_msg():
    return json.dumps({'id': 0, 'jsonrpc': '2.0', 'method': 'initialize', 'params': {'capabilities': {'workspace': {'didChangeConfiguration': {}}, 'textDocument': {}}, 'initializationOptions': None, 'processId': None, 'rootUri': pathlib.Path(__file__).parent.as_uri(), 'workspaceFolders': None}})