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
def on_startup(self, name: Optional[str]=None, verbose: Optional[bool]=None, _fx: Optional[Callable]=None, silenced: Optional[bool]=None, disabled: Optional[bool]=False, **kwargs):
    """
        Add a startup function to the worker queue.


        >> @Worker.on_startup("client")
        >> async def init_client():
        >>     return Client()

        >> ctx['client'] = Client


        >> @Worker.on_startup(_set_ctx = True)
        >> async def init_client(ctx: Dict[str, Any], **kwargs):
        >>   ctx['client'] = Client()
        >>   return ctx

        >> ctx['client'] = Client

        
        """
    if verbose is None:
        verbose = self.debug_enabled
    if _fx is not None:
        if disabled is True:
            return
        name = name or _fx.__name__
        self.tasks.startup_funcs[name] = (_fx, kwargs)
        if verbose:
            logger.info(f'Registered startup function {name}: {_fx}')
        if silenced is True:
            self.add_function_to_silenced(name)
        return

    def decorator(func: Callable):
        if disabled is True:
            return func
        func_name = name or func.__name__
        self.tasks.startup_funcs[func_name] = (func, kwargs)
        if verbose:
            logger.info(f'Registered startup function {func_name}: {func}')
        if silenced is True:
            self.add_function_to_silenced(name)
        return func
    return decorator