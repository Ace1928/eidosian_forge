from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
def release_snapshot(snapshot_id: SnapshotId) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, None]:
    """
    Releases layer snapshot captured by the back-end.

    :param snapshot_id: The id of the layer snapshot.
    """
    params: T_JSON_DICT = dict()
    params['snapshotId'] = snapshot_id.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'LayerTree.releaseSnapshot', 'params': params}
    json = (yield cmd_dict)