from __future__ import annotations
import re
import os
import sys
import logging
import typing
import traceback
import warnings
import pprint
import atexit as _atexit
import functools
import threading
from enum import Enum
from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from typing import Type, Union, Optional, Any, List, Dict, Tuple, Callable, Set, TYPE_CHECKING
@classmethod
def queue_logger_formatter(cls, record: Dict[str, Union[Dict[str, Any], Any]]) -> str:
    """
        Formats the log message for the queue.
        """
    _extra: Dict[str, Union[Dict[str, Any], Any]] = record.get('extra', {})
    if not record['extra'].get('worker_name'):
        record['extra']['worker_name'] = ''
    status = _extra.get('status')
    kind: str = _extra.get('kind')
    if status and isinstance(status, Enum):
        status = status.name
    kind_color = QUEUE_STATUS_COLORS.get(kind.lower(), FALLBACK_STATUS_COLOR)
    if '<' not in kind_color:
        kind_color = f'<{kind_color}>'
    extra = kind_color + '{extra[kind]}</>:'
    if _extra.get('queue_name'):
        queue_name_length = cls.get_extra_length('queue_name', _extra['queue_name'])
        extra += '<b><fg #006d77>{extra[queue_name]:<' + str(queue_name_length) + '}</></>:'
    if _extra.get('worker_name'):
        worker_name_length = cls.get_extra_length('worker_name', _extra['worker_name'])
        extra += '<fg #83c5be>{extra[worker_name]:<' + str(worker_name_length) + '}</>:'
    if _extra.get('job_id'):
        extra += '<fg #005f73>{extra[job_id]}</>'
    if status:
        status_color = QUEUE_STATUS_COLORS.get(status.lower(), FALLBACK_STATUS_COLOR)
        if '<' not in status_color:
            status_color = f'<{status_color}>'
        extra += f':{status_color}' + '{extra[status]}</>: '
    return extra