import abc
import asyncio
import datetime
import functools
import importlib
import json
import logging
import os
import pkgutil
from abc import ABCMeta, abstractmethod
from base64 import b64decode
from collections import namedtuple
from collections.abc import MutableMapping, Mapping, Sequence
from typing import Optional
import ray
import ray._private.ray_constants as ray_constants
import ray._private.services as services
from ray._raylet import GcsClient
from ray._private.utils import split_address
import aiosignal  # noqa: F401
import ray._private.protobuf_compat
from frozenlist import FrozenList  # noqa: F401
from ray._private.utils import binary_to_hex, check_dashboard_dependencies_installed
def async_loop_forever(interval_seconds, cancellable=False):

    def _wrapper(coro):

        @functools.wraps(coro)
        async def _looper(*args, **kwargs):
            while True:
                try:
                    await coro(*args, **kwargs)
                except asyncio.CancelledError as ex:
                    if cancellable:
                        logger.info(f'An async loop forever coroutine is cancelled {coro}.')
                        raise ex
                    else:
                        logger.exception(f'Can not cancel the async loop forever coroutine {coro}.')
                except Exception:
                    logger.exception(f'Error looping coroutine {coro}.')
                await asyncio.sleep(interval_seconds)
        return _looper
    return _wrapper