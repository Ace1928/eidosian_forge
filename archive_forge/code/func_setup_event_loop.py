from __future__ import annotations
import asyncio
import inspect
import json
import logging
import logging.config
import os
import socket
import ssl
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal
import click
from uvicorn._types import ASGIApplication
from uvicorn.importer import ImportFromStringError, import_from_string
from uvicorn.logging import TRACE_LOG_LEVEL
from uvicorn.middleware.asgi2 import ASGI2Middleware
from uvicorn.middleware.message_logger import MessageLoggerMiddleware
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
from uvicorn.middleware.wsgi import WSGIMiddleware
def setup_event_loop(self) -> None:
    loop_setup: Callable | None = import_from_string(LOOP_SETUPS[self.loop])
    if loop_setup is not None:
        loop_setup(use_subprocess=self.use_subprocess)