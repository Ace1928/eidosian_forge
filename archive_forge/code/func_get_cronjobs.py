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
def get_cronjobs(self, verbose: Optional[bool]=None) -> List['CronJob']:
    """
        Compiles all the worker cron functions
        that are enabled.
        WorkerCronFuncs = [CronJob(cron, cron="* * * * * */5")]
        """
    if verbose is None:
        verbose = self.debug_enabled
    from aiokeydb.v2.types.jobs import CronJob
    cronjobs = []
    for cron_op in self.tasks.cronjobs:
        if isinstance(cron_op, dict):
            silenced = cron_op.pop('silenced', None)
            cron_op = CronJob(**cron_op)
            if silenced is True:
                self.add_function_to_silenced(cron_op.function_name)
        if verbose:
            logger.info(f'Worker CronJob: {cron_op.function_name}: {cron_op.cron}')
        cronjobs.append(cron_op)
    return cronjobs