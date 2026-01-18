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
def create_page_profile_message(commands: list[Command], exec_time: int, prep_time: int, uncaught_exception: str | None=None) -> ForwardMsg:
    """Create and return the full PageProfile ForwardMsg."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    msg = ForwardMsg()
    page_profile = msg.page_profile
    page_profile.commands.extend(commands)
    page_profile.exec_time = exec_time
    page_profile.prep_time = prep_time
    page_profile.headless = config.get_option('server.headless')
    config_options: set[str] = set()
    if config._config_options:
        for option_name in config._config_options.keys():
            if not config.is_manually_set(option_name):
                continue
            config_option = config._config_options[option_name]
            if config_option.is_default:
                option_name = f'{option_name}:default'
            config_options.add(option_name)
    page_profile.config.extend(config_options)
    attributions: set[str] = {attribution for attribution in _ATTRIBUTIONS_TO_CHECK if attribution in sys.modules}
    page_profile.os = str(sys.platform)
    page_profile.timezone = str(time.tzname)
    page_profile.attributions.extend(attributions)
    if uncaught_exception:
        page_profile.uncaught_exception = uncaught_exception
    if (ctx := get_script_run_ctx()):
        page_profile.is_fragment_run = bool(ctx.current_fragment_id)
    return msg