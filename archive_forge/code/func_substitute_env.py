import asyncio
import atexit
import os
import string
import subprocess
from datetime import datetime, timezone
from tornado.ioloop import IOLoop
from tornado.queues import Queue
from tornado.websocket import WebSocketHandler
from traitlets import Bunch, Instance, Set, Unicode, UseEnum, observe
from traitlets.config import LoggingConfigurable
from . import stdio
from .schema import LANGUAGE_SERVER_SPEC
from .specs.utils import censored_spec
from .trait_types import Schema
from .types import SessionStatus
def substitute_env(self, env, base):
    final_env = base.copy()
    for key, value in env.items():
        final_env.update({key: string.Template(value).safe_substitute(base)})
    return final_env