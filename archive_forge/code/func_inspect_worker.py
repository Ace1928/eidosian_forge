from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import target
def inspect_worker(version_id: str) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    :param version_id:
    """
    params: T_JSON_DICT = dict()
    params['versionId'] = version_id
    cmd_dict: T_JSON_DICT = {'method': 'ServiceWorker.inspectWorker', 'params': params}
    json = (yield cmd_dict)