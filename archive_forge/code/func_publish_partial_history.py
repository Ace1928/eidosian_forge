import logging
import os
import sys
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterable, NewType, Optional, Tuple, Union
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_telemetry_pb2 as tpb
from wandb.sdk.artifacts.artifact import Artifact
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.staging import get_staging_dir
from wandb.sdk.lib import json_util as json
from wandb.util import (
from ..data_types.utils import history_dict_to_json, val_to_json
from ..lib.mailbox import MailboxHandle
from . import summary_record as sr
from .message_future import MessageFuture
def publish_partial_history(self, data: dict, user_step: int, step: Optional[int]=None, flush: Optional[bool]=None, publish_step: bool=True, run: Optional['Run']=None) -> None:
    run = run or self._run
    data = history_dict_to_json(run, data, step=user_step, ignore_copy_err=True)
    data.pop('_step', None)
    if '_timestamp' not in data:
        data['_timestamp'] = time.time()
    partial_history = pb.PartialHistoryRequest()
    for k, v in data.items():
        item = partial_history.item.add()
        item.key = k
        item.value_json = json_dumps_safer_history(v)
    if publish_step and step is not None:
        partial_history.step.num = step
    if flush is not None:
        partial_history.action.flush = flush
    self._publish_partial_history(partial_history)