from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import page
from . import target
def add_privacy_sandbox_enrollment_override(url: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Allows a site to use privacy sandbox features that require enrollment
    without the site actually being enrolled. Only supported on page targets.

    :param url:
    """
    params: T_JSON_DICT = dict()
    params['url'] = url
    cmd_dict: T_JSON_DICT = {'method': 'Browser.addPrivacySandboxEnrollmentOverride', 'params': params}
    json = (yield cmd_dict)