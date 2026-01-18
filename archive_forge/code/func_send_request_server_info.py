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
def send_request_server_info(self, record: 'Record') -> None:
    assert record.control.req_resp or record.control.mailbox_slot
    result = proto_util._result_from_record(record)
    result.response.server_info_response.local_info.CopyFrom(self.get_local_info())
    for message in self._server_messages:
        message_level = str(message.get('messageLevel'))
        message_level_sanitized = int(printer.INFO if not message_level.isdigit() else message_level)
        result.response.server_info_response.server_messages.item.append(wandb_internal_pb2.ServerMessage(utf_text=message.get('utfText', ''), plain_text=message.get('plainText', ''), html_text=message.get('htmlText', ''), type=message.get('messageType', ''), level=message_level_sanitized))
    self._respond_result(result)