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
def send_run(run: ImporterRun, *, extra_arts: Optional[Iterable[Artifact]]=None, extra_used_arts: Optional[Iterable[Artifact]]=None, config: Optional[SendManagerConfig]=None, overrides: Optional[Dict[str, Any]]=None, settings_override: Optional[Dict[str, Any]]=None) -> None:
    if config is None:
        config = SendManagerConfig()
    if overrides:
        for k, v in overrides.items():
            setattr(run, k, lambda v=v: v)
    rm = RecordMaker(run)
    root_dir = rm.run_dir
    settings = _make_settings(root_dir, settings_override)
    sm_record_q = queue.Queue()
    result_q = queue.Queue()
    interface = InterfaceQueue(record_q=sm_record_q)
    context_keeper = context.ContextKeeper()
    sm = AlternateSendManager(settings, sm_record_q, result_q, interface, context_keeper)
    if extra_arts or extra_used_arts:
        records = rm.make_artifacts_only_records(extra_arts, extra_used_arts)
    else:
        records = rm.make_records(config)
    for r in records:
        logger.debug(f'Sending r={r!r}')
        sm.send(r)
    sm.finish()