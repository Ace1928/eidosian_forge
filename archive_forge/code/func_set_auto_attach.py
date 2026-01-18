from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def set_auto_attach(auto_attach: bool, wait_for_debugger_on_start: bool, flatten: typing.Optional[bool]=None, filter_: typing.Optional[TargetFilter]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Controls whether to automatically attach to new targets which are considered to be related to
    this one. When turned on, attaches to all existing related targets as well. When turned off,
    automatically detaches from all currently attached targets.
    This also clears all targets added by ``autoAttachRelated`` from the list of targets to watch
    for creation of related targets.

    :param auto_attach: Whether to auto-attach to related targets.
    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.
    :param flatten: **(EXPERIMENTAL)** *(Optional)* Enables "flat" access to the session via specifying sessionId attribute in the commands. We plan to make this the default, deprecate non-flattened mode, and eventually retire it. See crbug.com/991325.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.
    """
    params: T_JSON_DICT = dict()
    params['autoAttach'] = auto_attach
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if flatten is not None:
        params['flatten'] = flatten
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.setAutoAttach', 'params': params}
    json = (yield cmd_dict)