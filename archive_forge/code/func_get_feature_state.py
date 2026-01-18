from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def get_feature_state(feature_state: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, bool]:
    """
    Returns information about the feature state.

    :param feature_state:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['featureState'] = feature_state
    cmd_dict: T_JSON_DICT = {'method': 'SystemInfo.getFeatureState', 'params': params}
    json = (yield cmd_dict)
    return bool(json['featureEnabled'])