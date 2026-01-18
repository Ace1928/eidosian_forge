from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def override_quota_for_origin(origin: str, quota_size: typing.Optional[float]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Override quota for the specified origin

    **EXPERIMENTAL**

    :param origin: Security origin.
    :param quota_size: *(Optional)* The quota size (in bytes) to override the original quota with. If this is called multiple times, the overridden quota will be equal to the quotaSize provided in the final call. If this is called without specifying a quotaSize, the quota will be reset to the default value for the specified origin. If this is called multiple times with different origins, the override will be maintained for each origin until it is disabled (called without a quotaSize).
    """
    params: T_JSON_DICT = dict()
    params['origin'] = origin
    if quota_size is not None:
        params['quotaSize'] = quota_size
    cmd_dict: T_JSON_DICT = {'method': 'Storage.overrideQuotaForOrigin', 'params': params}
    json = (yield cmd_dict)