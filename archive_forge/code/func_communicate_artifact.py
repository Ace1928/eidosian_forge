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
def communicate_artifact(self, run: 'Run', artifact: 'Artifact', aliases: Iterable[str], history_step: Optional[int]=None, is_user_created: bool=False, use_after_commit: bool=False, finalize: bool=True) -> MessageFuture:
    proto_run = self._make_run(run)
    proto_artifact = self._make_artifact(artifact)
    proto_artifact.run_id = proto_run.run_id
    proto_artifact.project = proto_run.project
    proto_artifact.entity = proto_run.entity
    proto_artifact.user_created = is_user_created
    proto_artifact.use_after_commit = use_after_commit
    proto_artifact.finalize = finalize
    for alias in aliases:
        proto_artifact.aliases.append(alias)
    log_artifact = pb.LogArtifactRequest()
    log_artifact.artifact.CopyFrom(proto_artifact)
    if history_step is not None:
        log_artifact.history_step = history_step
    log_artifact.staging_dir = get_staging_dir()
    resp = self._communicate_artifact(log_artifact)
    return resp