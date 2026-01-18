from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import network
def handle_certificate_error(event_id: int, action: CertificateErrorAction) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Handles a certificate error that fired a certificateError event.

    :param event_id: The ID of the event.
    :param action: The action to take on the certificate error.
    """
    params: T_JSON_DICT = dict()
    params['eventId'] = event_id
    params['action'] = action.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'Security.handleCertificateError', 'params': params}
    json = (yield cmd_dict)