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
def post(cls, path, **kwargs):
    return cls._register_route(hdrs.METH_POST, path, **kwargs)