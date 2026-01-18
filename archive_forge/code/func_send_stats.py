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
def send_stats(self, record: 'Record') -> None:
    stats = record.stats
    if stats.stats_type != wandb_internal_pb2.StatsRecord.StatsType.SYSTEM:
        return
    if not self._fs:
        return
    if not self._run:
        return
    now_us = stats.timestamp.ToMicroseconds()
    start_us = self._run.start_time.ToMicroseconds()
    d = dict()
    for item in stats.item:
        d[item.key] = json.loads(item.value_json)
    row: Dict[str, Any] = dict(system=d)
    self._flatten(row)
    row['_wandb'] = True
    row['_timestamp'] = now_us / 1000000.0
    row['_runtime'] = (now_us - start_us) / 1000000.0
    self._fs.push(filenames.EVENTS_FNAME, json.dumps(row))