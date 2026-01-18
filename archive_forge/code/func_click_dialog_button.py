from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def click_dialog_button(dialog_id: str, dialog_button: DialogButton) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param dialog_id:
    :param dialog_button:
    """
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    params['dialogButton'] = dialog_button.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'FedCm.clickDialogButton', 'params': params}
    json = (yield cmd_dict)