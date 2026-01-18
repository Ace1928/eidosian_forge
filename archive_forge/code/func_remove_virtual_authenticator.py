from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def remove_virtual_authenticator(authenticator_id: AuthenticatorId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Removes the given authenticator.

    :param authenticator_id:
    """
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'WebAuthn.removeVirtualAuthenticator', 'params': params}
    json = (yield cmd_dict)