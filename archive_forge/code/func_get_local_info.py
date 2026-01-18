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
def get_local_info(self) -> 'LocalInfo':
    """Queries the server to get the local version information.

        First, we perform an introspection, if it returns empty we deduce that the
        docker image is out-of-date. Otherwise, we use the returned values to deduce the
        state of the local server.
        """
    local_info = wandb_internal_pb2.LocalInfo()
    if self._settings._offline:
        local_info.out_of_date = False
        return local_info
    latest_local_version = 'latest'
    server_info = self.get_server_info()
    latest_local_version_info = server_info.get('latestLocalVersionInfo', {})
    if latest_local_version_info is None:
        local_info.out_of_date = False
    else:
        local_info.out_of_date = latest_local_version_info.get('outOfDate', True)
        local_info.version = latest_local_version_info.get('latestVersionString', latest_local_version)
    return local_info