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
def deliver_download_artifact(self, artifact_id: str, download_root: str, allow_missing_references: bool) -> MailboxHandle:
    download_artifact = pb.DownloadArtifactRequest()
    download_artifact.artifact_id = artifact_id
    download_artifact.download_root = download_root
    download_artifact.allow_missing_references = allow_missing_references
    resp = self._deliver_download_artifact(download_artifact)
    return resp