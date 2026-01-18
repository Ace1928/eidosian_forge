from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def set_interest_group_auction_tracking(enable: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Enables/Disables issuing of interestGroupAuctionEvent events.

    **EXPERIMENTAL**

    :param enable:
    """
    params: T_JSON_DICT = dict()
    params['enable'] = enable
    cmd_dict: T_JSON_DICT = {'method': 'Storage.setInterestGroupAuctionTracking', 'params': params}
    json = (yield cmd_dict)