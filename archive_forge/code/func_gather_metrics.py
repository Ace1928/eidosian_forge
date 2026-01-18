from __future__ import annotations
import contextlib
import inspect
import os
import sys
import threading
import time
import uuid
from collections.abc import Sized
from functools import wraps
from typing import Any, Callable, Final, TypeVar, cast, overload
from streamlit import config, util
from streamlit.logger import get_logger
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.proto.PageProfile_pb2 import Argument, Command
def gather_metrics(name: str, func: F | None=None) -> Callable[[F], F] | F:
    """Function decorator to add telemetry tracking to commands.

    Parameters
    ----------
    func : callable
    The function to track for telemetry.

    name : str or None
    Overwrite the function name with a custom name that is used for telemetry tracking.

    Example
    -------
    >>> @st.gather_metrics
    ... def my_command(url):
    ...     return url

    >>> @st.gather_metrics(name="custom_name")
    ... def my_command(url):
    ...     return url
    """
    if not name:
        _LOGGER.warning('gather_metrics: name is empty')
        name = 'undefined'
    if func is None:

        def wrapper(f: F) -> F:
            return gather_metrics(name=name, func=f)
        return wrapper
    else:
        non_optional_func = func

    @wraps(non_optional_func)
    def wrapped_func(*args, **kwargs):
        from timeit import default_timer as timer
        exec_start = timer()
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        from streamlit.runtime.scriptrunner.script_runner import RerunException
        ctx = get_script_run_ctx(suppress_warning=True)
        tracking_activated = ctx is not None and ctx.gather_usage_stats and (not ctx.command_tracking_deactivated) and (len(ctx.tracked_commands) < _MAX_TRACKED_COMMANDS)
        command_telemetry: Command | None = None
        if ctx and tracking_activated:
            try:
                command_telemetry = _get_command_telemetry(non_optional_func, name, *args, **kwargs)
                if command_telemetry.name not in ctx.tracked_commands_counter or ctx.tracked_commands_counter[command_telemetry.name] < _MAX_TRACKED_PER_COMMAND:
                    ctx.tracked_commands.append(command_telemetry)
                ctx.tracked_commands_counter.update([command_telemetry.name])
                ctx.command_tracking_deactivated = True
            except Exception as ex:
                _LOGGER.debug('Failed to collect command telemetry', exc_info=ex)
        try:
            result = non_optional_func(*args, **kwargs)
        except RerunException as ex:
            if tracking_activated and command_telemetry:
                command_telemetry.time = to_microseconds(timer() - exec_start)
            raise ex
        finally:
            if ctx:
                ctx.command_tracking_deactivated = False
        if tracking_activated and command_telemetry:
            command_telemetry.time = to_microseconds(timer() - exec_start)
        return result
    with contextlib.suppress(AttributeError):
        wrapped_func.__dict__.update(non_optional_func.__dict__)
        wrapped_func.__signature__ = inspect.signature(non_optional_func)
    return cast(F, wrapped_func)