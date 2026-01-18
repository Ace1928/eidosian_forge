from __future__ import annotations
import os
import time
import uuid
import random
import typing
import logging
import traceback
import contextlib
import contextvars
import asyncio
import functools
import inspect
from concurrent import futures
from lazyops.utils.helpers import import_function, timer, build_batches
from lazyops.utils.ahelpers import amap_v2 as concurrent_map
def isclassmethod(method):
    bound_to = getattr(method, '__self__', None)
    if not isinstance(bound_to, type):
        return False
    name = method.__name__
    for cls in bound_to.__mro__:
        descriptor = vars(cls).get(name)
        if descriptor is not None:
            return isinstance(descriptor, classmethod)
    return False