from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
def set_credential_properties(authenticator_id: AuthenticatorId, credential_id: str, backup_eligibility: typing.Optional[bool]=None, backup_state: typing.Optional[bool]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Allows setting credential properties.
    https://w3c.github.io/webauthn/#sctn-automation-set-credential-properties

    :param authenticator_id:
    :param credential_id:
    :param backup_eligibility: *(Optional)*
    :param backup_state: *(Optional)*
    """
    params: T_JSON_DICT = dict()
    params['authenticatorId'] = authenticator_id.to_json()
    params['credentialId'] = credential_id
    if backup_eligibility is not None:
        params['backupEligibility'] = backup_eligibility
    if backup_state is not None:
        params['backupState'] = backup_state
    cmd_dict: T_JSON_DICT = {'method': 'WebAuthn.setCredentialProperties', 'params': params}
    json = (yield cmd_dict)