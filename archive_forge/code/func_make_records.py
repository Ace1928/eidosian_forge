import json
import logging
import math
import os
import queue
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import numpy as np
from google.protobuf.json_format import ParseDict
from tenacity import retry, stop_after_attempt, wait_random_exponential
from wandb import Artifact
from wandb.proto import wandb_internal_pb2 as pb
from wandb.proto import wandb_settings_pb2
from wandb.proto import wandb_telemetry_pb2 as telem_pb
from wandb.sdk.interface.interface import file_policy_to_enum
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import context
from wandb.sdk.internal.sender import SendManager
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.util import coalesce, recursive_cast_dictlike_to_dict
from .protocols import ImporterRun
def make_records(self, config: SendManagerConfig) -> Iterable[pb.Record]:
    """Make all the records that constitute a run."""
    yield self._make_run_record()
    yield self._make_telem_record()
    include_artifacts = config.log_artifacts or config.use_artifacts
    yield self._make_files_record(include_artifacts, config.files, config.media, config.code)
    if config.use_artifacts:
        if (used_artifacts := self.run.used_artifacts()) is not None:
            for artifact in used_artifacts:
                yield self._make_artifact_record(artifact, use_artifact=True)
    if config.log_artifacts:
        if (artifacts := self.run.artifacts()) is not None:
            for artifact in artifacts:
                yield self._make_artifact_record(artifact)
    if config.history:
        yield from self._make_history_records()
    if config.summary:
        yield self._make_summary_record()
    if config.terminal_output:
        if (lines := self.run.logs()) is not None:
            for line in lines:
                yield self._make_output_record(line)