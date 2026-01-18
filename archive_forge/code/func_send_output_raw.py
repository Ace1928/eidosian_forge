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
def send_output_raw(self, record: 'Record') -> None:
    if not self._fs:
        return
    out = record.output_raw
    stream: StreamLiterals = 'stdout'
    if out.output_type == wandb_internal_pb2.OutputRawRecord.OutputType.STDERR:
        stream = 'stderr'
    line = out.line
    output_raw = self._output_raw_streams.get(stream)
    if not output_raw:
        output_raw = _OutputRawStream(stream=stream, sm=self)
        self._output_raw_streams[stream] = output_raw
        if not self._output_raw_file:
            output_log_path = os.path.join(self._settings.files_dir, filenames.OUTPUT_FNAME)
            output_raw_file = None
            try:
                output_raw_file = filesystem.CRDedupedFile(open(output_log_path, 'wb'))
            except OSError as e:
                logger.warning(f'could not open output_raw_file: {e}')
            if output_raw_file:
                self._output_raw_file = output_raw_file
        output_raw.start()
    output_raw._queue.put(line)