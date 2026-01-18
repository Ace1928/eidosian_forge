from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def get_interest_group_details(owner_origin: str, name: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, InterestGroupDetails]:
    """
    Gets details for a named interest group.

    **EXPERIMENTAL**

    :param owner_origin:
    :param name:
    :returns: 
    """
    params: T_JSON_DICT = dict()
    params['ownerOrigin'] = owner_origin
    params['name'] = name
    cmd_dict: T_JSON_DICT = {'method': 'Storage.getInterestGroupDetails', 'params': params}
    json = (yield cmd_dict)
    return InterestGroupDetails.from_json(json['details'])