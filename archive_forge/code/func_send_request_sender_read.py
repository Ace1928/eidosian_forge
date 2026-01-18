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
def send_request_sender_read(self, record: 'Record') -> None:
    if self._ds is None:
        self._ds = datastore.DataStore()
        self._ds.open_for_scan(self._settings.sync_file)
    start_offset = record.request.sender_read.start_offset
    final_offset = record.request.sender_read.final_offset
    self._ds.seek(start_offset)
    current_end_offset = 0
    while current_end_offset < final_offset:
        data = self._ds.scan_data()
        assert data
        current_end_offset = self._ds.get_offset()
        send_record = wandb_internal_pb2.Record()
        send_record.ParseFromString(data)
        self._update_end_offset(current_end_offset)
        self.send(send_record)
        self.debounce()
    self._maybe_report_status(always=True)