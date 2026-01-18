from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import page
def set_discover_targets(discover: bool, filter_: typing.Optional[TargetFilter]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Controls whether to discover available targets and notify via
    ``targetCreated/targetInfoChanged/targetDestroyed`` events.

    :param discover: Whether to discover available targets.
    :param filter_: **(EXPERIMENTAL)** *(Optional)* Only targets matching filter will be attached. If ```discover```` is false, ````filter``` must be omitted or empty.
    """
    params: T_JSON_DICT = dict()
    params['discover'] = discover
    if filter_ is not None:
        params['filter'] = filter_.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Target.setDiscoverTargets', 'params': params}
    json = (yield cmd_dict)