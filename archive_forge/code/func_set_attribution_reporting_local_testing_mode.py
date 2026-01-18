from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import browser
from . import network
from . import page
def set_attribution_reporting_local_testing_mode(enabled: bool) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    https://wicg.github.io/attribution-reporting-api/

    **EXPERIMENTAL**

    :param enabled: If enabled, noise is suppressed and reports are sent immediately.
    """
    params: T_JSON_DICT = dict()
    params['enabled'] = enabled
    cmd_dict: T_JSON_DICT = {'method': 'Storage.setAttributionReportingLocalTestingMode', 'params': params}
    json = (yield cmd_dict)