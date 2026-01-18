from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def select_account(dialog_id: str, account_index: int) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param dialog_id:
    :param account_index:
    """
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['accountIndex'] = account_index
    cmd_dict: T_JSON_DICT = {'method': 'FedCm.selectAccount', 'params': params}
    json = (yield cmd_dict)