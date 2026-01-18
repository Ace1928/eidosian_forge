from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def get_usage_and_quota(origin: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.Tuple[float, float, bool, typing.List[UsageForType]]]:
    """
    Returns usage and quota in bytes.

    :param origin: Security origin.
    :returns: A tuple with the following items:

        0. **usage** - Storage usage (bytes).
        1. **quota** - Storage quota (bytes).
        2. **overrideActive** - Whether or not the origin has an active storage quota override
        3. **usageBreakdown** - Storage usage per type (bytes).
    """
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    cmd_dict: T_JSON_DICT = {'method': 'Storage.getUsageAndQuota', 'params': params}
    json = (yield cmd_dict)
    return (float(json['usage']), float(json['quota']), bool(json['overrideActive']), [UsageForType.from_json(i) for i in json['usageBreakdown']])