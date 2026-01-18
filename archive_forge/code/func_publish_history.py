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
def publish_history(self, data: dict, step: Optional[int]=None, run: Optional['Run']=None, publish_step: bool=True) -> None:
    run = run or self._run
    data = history_dict_to_json(run, data, step=step)
    history = pb.HistoryRecord()
    if publish_step:
        assert step is not None
        history.step.num = step
    data.pop('_step', None)
    for k, v in data.items():
        item = history.item.add()
        item.key = k
        item.value_json = json_dumps_safer_history(v)
    self._publish_history(history)