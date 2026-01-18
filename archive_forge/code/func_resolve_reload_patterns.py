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
def resolve_reload_patterns(patterns_list: list[str], directories_list: list[str]) -> tuple[list[str], list[Path]]:
    directories: list[Path] = list(set(map(Path, directories_list.copy())))
    patterns: list[str] = patterns_list.copy()
    current_working_directory = Path.cwd()
    for pattern in patterns_list:
        if pattern == '.*':
            continue
        patterns.append(pattern)
        if is_dir(Path(pattern)):
            directories.append(Path(pattern))
        else:
            for match in current_working_directory.glob(pattern):
                if is_dir(match):
                    directories.append(match)
    directories = list(set(directories))
    directories = list(map(Path, directories))
    directories = list(map(lambda x: x.resolve(), directories))
    directories = list({reload_path for reload_path in directories if is_dir(reload_path)})
    children = []
    for j in range(len(directories)):
        for k in range(j + 1, len(directories)):
            if directories[j] in directories[k].parents:
                children.append(directories[k])
            elif directories[k] in directories[j].parents:
                children.append(directories[j])
    directories = list(set(directories).difference(set(children)))
    return (list(set(patterns)), directories)