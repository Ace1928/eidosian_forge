from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def dismiss_dialog(dialog_id: str, trigger_cooldown: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param dialog_id:
    :param trigger_cooldown: *(Optional)*
    """
    params: T_JSON_DICT = dict()
    params['dialogId'] = dialog_id
    if trigger_cooldown is not None:
        params['triggerCooldown'] = trigger_cooldown
    cmd_dict: T_JSON_DICT = {'method': 'FedCm.dismissDialog', 'params': params}
    json = (yield cmd_dict)