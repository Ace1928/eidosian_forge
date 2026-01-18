from __future__ import annotations
import logging # isort:skip
import gc
import os
from pprint import pformat
from types import ModuleType
from typing import (
from urllib.parse import urljoin
from tornado.ioloop import PeriodicCallback
from tornado.web import Application as TornadoApplication, StaticFileHandler
from tornado.websocket import WebSocketClosedError
from ..application import Application
from ..document import Document
from ..model import Model
from ..resources import Resources
from ..settings import settings
from ..util.dependencies import import_optional
from ..util.strings import format_docstring
from ..util.tornado import fixup_windows_event_loop_policy
from .auth_provider import NullAuth
from .connection import ServerConnection
from .contexts import ApplicationContext
from .session import ServerSession
from .urls import per_app_patterns, toplevel_patterns
from .views.ico_handler import IcoHandler
from .views.root_handler import RootHandler
from .views.static_handler import StaticHandler
from .views.ws import WSHandler
def get_sessions(self, app_path: str) -> list[ServerSession]:
    """ Gets all currently active sessions for an application.

        Args:
            app_path (str) :
                The configured application path for the application to return
                sessions for.

        Returns:
            list[ServerSession]

        """
    if app_path not in self._applications:
        raise ValueError('Application %s does not exist on this server' % app_path)
    return list(self._applications[app_path].sessions)