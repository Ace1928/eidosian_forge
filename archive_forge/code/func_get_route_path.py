from __future__ import annotations
import asyncio
import functools
import re
import sys
import typing
from contextlib import contextmanager
from starlette.types import Scope
def get_route_path(scope: Scope) -> str:
    root_path = scope.get('root_path', '')
    route_path = re.sub('^' + root_path, '', scope['path'])
    return route_path