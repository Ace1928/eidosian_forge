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
def send_request_login(self, record: 'Record') -> None:
    self._api.reauth()
    viewer = self.get_viewer_info()
    server_info = self.get_server_info()
    if server_info:
        logger.info(f'Login server info: {server_info}')
    self._entity = viewer.get('entity')
    if record.control.req_resp:
        result = proto_util._result_from_record(record)
        if self._entity:
            result.response.login_response.active_entity = self._entity
        self._respond_result(result)