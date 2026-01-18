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
def send_alert(self, record: 'Record') -> None:
    from wandb.util import parse_version
    alert = record.alert
    max_cli_version = self._max_cli_version()
    if max_cli_version is None or parse_version(max_cli_version) < parse_version('0.10.9'):
        logger.warning("This W&B server doesn't support alerts, have your administrator install wandb/local >= 0.9.31")
    else:
        try:
            self._api.notify_scriptable_run_alert(title=alert.title, text=alert.text, level=alert.level, wait_duration=alert.wait_duration)
        except Exception as e:
            logger.error(f'send_alert: failed for alert {alert.title!r}: {e}')