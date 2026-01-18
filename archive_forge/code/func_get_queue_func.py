import os
import re
import json
import socket
import contextlib
import functools
from lazyops.utils.helpers import is_coro_func
from lazyops.utils.logs import default_logger as logger
from typing import Optional, Dict, Any, Union, Callable, List, Tuple, TYPE_CHECKING
from aiokeydb.v2.types import BaseSettings, validator, lazyproperty, KeyDBUri
from aiokeydb.v2.types.static import TaskType
from aiokeydb.v2.serializers import SerializerType
from aiokeydb.v2.utils.queue import run_in_executor
from aiokeydb.v2.utils.cron import validate_cron_schedule
def get_queue_func(self, queue_func: Optional[Union[Callable, 'TaskQueue']]=None) -> 'TaskQueue':
    """
        Gets the queue function to use.
        """
    queue_func = queue_func or self.tasks.queue_func
    return queue_func() if callable(queue_func) else queue_func