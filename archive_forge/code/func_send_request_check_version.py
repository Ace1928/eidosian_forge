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
def send_request_check_version(self, record: 'Record') -> None:
    assert record.control.req_resp or record.control.mailbox_slot
    result = proto_util._result_from_record(record)
    current_version = record.request.check_version.current_version or wandb.__version__
    messages = update.check_available(current_version)
    if messages:
        upgrade_message = messages.get('upgrade_message')
        if upgrade_message:
            result.response.check_version_response.upgrade_message = upgrade_message
        yank_message = messages.get('yank_message')
        if yank_message:
            result.response.check_version_response.yank_message = yank_message
        delete_message = messages.get('delete_message')
        if delete_message:
            result.response.check_version_response.delete_message = delete_message
    self._respond_result(result)