import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from queue import Queue
from typing import (
import requests
import wandb
from wandb import util
from wandb.errors import CommError, UsageError
from wandb.errors.util import ProtobufErrorHandler
from wandb.filesync.dir_watcher import DirWatcher
from wandb.proto import wandb_internal_pb2
from wandb.sdk.artifacts.artifact_saver import ArtifactSaver
from wandb.sdk.interface import interface
from wandb.sdk.interface.interface_queue import InterfaceQueue
from wandb.sdk.internal import (
from wandb.sdk.internal.file_pusher import FilePusher
from wandb.sdk.internal.job_builder import JobBuilder
from wandb.sdk.internal.settings_static import SettingsStatic
from wandb.sdk.lib import (
from wandb.sdk.lib.mailbox import ContextCancelledError
from wandb.sdk.lib.proto_util import message_to_dict
def send_metric(self, record: 'Record') -> None:
    metric = record.metric
    if metric.glob_name:
        logger.warning("Seen metric with glob (shouldn't happen)")
        return
    old_metric = self._config_metric_dict.get(metric.name, wandb_internal_pb2.MetricRecord())
    if metric._control.overwrite:
        old_metric.CopyFrom(metric)
    else:
        old_metric.MergeFrom(metric)
    self._config_metric_dict[metric.name] = old_metric
    metric = old_metric
    if metric.step_metric:
        find_step_idx = self._config_metric_index_dict.get(metric.step_metric)
        if find_step_idx is not None:
            rec = wandb_internal_pb2.Record()
            rec.metric.CopyFrom(metric)
            metric = rec.metric
            metric.ClearField('step_metric')
            metric.step_metric_index = find_step_idx + 1
    md: Dict[int, Any] = proto_util.proto_encode_to_dict(metric)
    find_idx = self._config_metric_index_dict.get(metric.name)
    if find_idx is not None:
        self._config_metric_pbdict_list[find_idx] = md
    else:
        next_idx = len(self._config_metric_pbdict_list)
        self._config_metric_pbdict_list.append(md)
        self._config_metric_index_dict[metric.name] = next_idx
    self._update_config()