from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def get_browser_command_line() -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[str]]:
    """
    Returns the command line switches for the browser process if, and only if
    --enable-automation is on the commandline.

    **EXPERIMENTAL**

    :returns: Commandline parameters
    """
    cmd_dict: T_JSON_DICT = {'method': 'Browser.getBrowserCommandLine'}
    json = (yield cmd_dict)
    return [str(i) for i in json['arguments']]