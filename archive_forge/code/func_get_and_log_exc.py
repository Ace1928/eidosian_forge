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
def get_and_log_exc(job: typing.Optional['Job']=None, func: typing.Optional[typing.Union[str, typing.Callable]]=None, chain: typing.Optional[bool]=True, limit: typing.Optional[int]=TRACE_CHAIN_LIMIT):
    error = traceback.format_exc(chain=chain, limit=limit)
    err_msg = f'node={get_hostname()}, {error}'
    if func:
        err_msg = f'func={get_func_name(func)}, {err_msg}'
    elif job:
        err_msg = f'job={job.short_repr}, {err_msg}'
    logger.error(err_msg)
    return error