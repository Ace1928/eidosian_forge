from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_user_verified(authenticator_id: AuthenticatorId, is_user_verified: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Sets whether User Verification succeeds or fails for an authenticator.
    The default is true.

    :param authenticator_id:
    :param is_user_verified:
    """
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['isUserVerified'] = is_user_verified
    cmd_dict: T_JSON_DICT = {'method': 'WebAuthn.setUserVerified', 'params': params}
    json = (yield cmd_dict)