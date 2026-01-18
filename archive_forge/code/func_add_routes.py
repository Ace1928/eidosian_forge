import abc
import asyncio
import base64
import hashlib
import inspect
import keyword
import os
import re
import warnings
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from types import MappingProxyType
from typing import (
from yarl import URL, __version__ as yarl_version  # type: ignore[attr-defined]
from . import hdrs
from .abc import AbstractMatchInfo, AbstractRouter, AbstractView
from .helpers import DEBUG
from .http import HttpVersion11
from .typedefs import Handler, PathLike
from .web_exceptions import (
from .web_fileresponse import FileResponse
from .web_request import Request
from .web_response import Response, StreamResponse
from .web_routedef import AbstractRouteDef
def add_routes(self, routes: Iterable[AbstractRouteDef]) -> List[AbstractRoute]:
    """Append routes to route table.

        Parameter should be a sequence of RouteDef objects.

        Returns a list of registered AbstractRoute instances.
        """
    registered_routes = []
    for route_def in routes:
        registered_routes.extend(route_def.register(self))
    return registered_routes