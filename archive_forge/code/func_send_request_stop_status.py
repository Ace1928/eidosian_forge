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
def send_request_stop_status(self, record: 'Record') -> None:
    result = proto_util._result_from_record(record)
    status_resp = result.response.stop_status_response
    status_resp.run_should_stop = False
    if self._entity and self._project and self._run and self._run.run_id:
        try:
            status_resp.run_should_stop = self._api.check_stop_requested(self._project, self._entity, self._run.run_id)
        except Exception as e:
            logger.warning('Failed to check stop requested status: %s', e)
    self._respond_result(result)