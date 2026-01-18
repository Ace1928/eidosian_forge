import asyncio
import collections
import functools
import inspect
import json
import logging
import os
import time
import traceback
from collections import namedtuple
from typing import Any, Callable
from aiohttp.web import Response
import ray
import ray.dashboard.consts as dashboard_consts
from ray._private.ray_constants import RAY_INTERNAL_DASHBOARD_NAMESPACE, env_bool
from ray.dashboard.optional_deps import PathLike, RouteDef, aiohttp, hdrs
from ray.dashboard.utils import CustomEncoder, to_google_style
@classmethod
def bound_routes(cls):
    bound_items = []
    for r in cls._routes._items:
        if isinstance(r, RouteDef):
            route_method = getattr(r.handler, '__route_method__')
            route_path = getattr(r.handler, '__route_path__')
            instance = cls._bind_map[route_method][route_path].instance
            if instance is not None:
                bound_items.append(r)
        else:
            bound_items.append(r)
    routes = aiohttp.web.RouteTableDef()
    routes._items = bound_items
    return routes