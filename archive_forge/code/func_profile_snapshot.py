from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
def profile_snapshot(snapshot_id: SnapshotId, min_repeat_count: typing.Optional[int]=None, min_duration: typing.Optional[float]=None, clip_rect: typing.Optional[dom.Rect]=None) -> typing.Generator[T_JSON_DICT, T_JSON_DICT, typing.List[PaintProfile]]:
    """
    :param snapshot_id: The id of the layer snapshot.
    :param min_repeat_count: *(Optional)* The maximum number of times to replay the snapshot (1, if not specified).
    :param min_duration: *(Optional)* The minimum duration (in seconds) to replay the snapshot.
    :param clip_rect: *(Optional)* The clip rectangle to apply when replaying the snapshot.
    :returns: The array of paint profiles, one per run.
    """
    params: T_JSON_DICT = dict()
    params['snapshotId'] = snapshot_id.to_json()
    if min_repeat_count is not None:
        params['minRepeatCount'] = min_repeat_count
    if min_duration is not None:
        params['minDuration'] = min_duration
    if clip_rect is not None:
        params['clipRect'] = clip_rect.to_json()
    cmd_dict: T_JSON_DICT = {'method': 'LayerTree.profileSnapshot', 'params': params}
    json = (yield cmd_dict)
    return [PaintProfile.from_json(i) for i in json['timings']]