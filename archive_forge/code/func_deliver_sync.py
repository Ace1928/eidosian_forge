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
def deliver_sync(self, start_offset: int, final_offset: int, entity: Optional[str]=None, project: Optional[str]=None, run_id: Optional[str]=None, skip_output_raw: Optional[bool]=None) -> MailboxHandle:
    sync = pb.SyncRequest(start_offset=start_offset, final_offset=final_offset)
    if entity:
        sync.overwrite.entity = entity
    if project:
        sync.overwrite.project = project
    if run_id:
        sync.overwrite.run_id = run_id
    if skip_output_raw:
        sync.skip.output_raw = skip_output_raw
    return self._deliver_sync(sync)