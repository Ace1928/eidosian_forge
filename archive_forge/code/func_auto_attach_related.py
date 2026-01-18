from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def auto_attach_related(target_id: TargetID, wait_for_debugger_on_start: bool, filter_: typing.Optional[TargetFilter]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Adds the specified target to the list of targets that will be monitored for any related target
    creation (such as child frames, child workers and new versions of service worker) and reported
    through ``attachedToTarget``. The specified target is also auto-attached.
    This cancels the effect of any previous ``setAutoAttach`` and is also cancelled by subsequent
    ``setAutoAttach``. Only available at the Browser target.

    **EXPERIMENTAL**

    :param target_id:
    :param wait_for_debugger_on_start: Whether to pause new targets when attaching to them. Use ```Runtime.runIfWaitingForDebugger``` to run paused targets.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached.
    """
    params: T_JSON_DICT = dict()
    params['targetId'] = target_id.to_json()
    params['waitForDebuggerOnStart'] = wait_for_debugger_on_start
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.autoAttachRelated', 'params': params}
    json = (yield cmd_dict)