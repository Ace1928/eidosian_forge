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
def publish_use_artifact(self, artifact: 'Artifact') -> None:
    assert artifact.id is not None, 'Artifact must have an id'
    use_artifact = pb.UseArtifactRecord(id=artifact.id, type=artifact.type, name=artifact.name)
    if '_partial' in artifact.metadata:
        job_info = {}
        try:
            path = artifact.get_entry('wandb-job.json').download()
            with open(path) as f:
                job_info = json.load(f)
        except Exception as e:
            logger.warning(f'Failed to download partial job info from artifact {artifact}, : {e}')
        use_artifact = self._make_proto_use_artifact(use_artifact=use_artifact, job_name=artifact.name, job_info=job_info, metadata=artifact.metadata)
    self._publish_use_artifact(use_artifact)