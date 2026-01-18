from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import debugger
from . import dom
from . import emulation
from . import io
from . import network
from . import runtime
def set_intercept_file_chooser_dialog(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Intercept file chooser requests and transfer control to protocol clients.
    When file chooser interception is enabled, native file chooser dialog is not shown.
    Instead, a protocol event ``Page.fileChooserOpened`` is emitted.

    :param enabled:
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Page.setInterceptFileChooserDialog', 'params': params}
    json = (yield cmd_dict)