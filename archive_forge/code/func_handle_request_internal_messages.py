import json
import logging
import math
import numbers
import time
from collections import defaultdict
from queue import Queue
from threading import Event
from typing import (
from wandb.proto.wandb_internal_pb2 import (
from ..interface.interface_queue import InterfaceQueue
from ..lib import handler_util, proto_util, tracelog, wburls
from . import context, sample, tb_watcher
from .settings_static import SettingsStatic
from .system.system_monitor import SystemMonitor
def handle_request_internal_messages(self, record: Record) -> None:
    result = proto_util._result_from_record(record)
    result.response.internal_messages_response.messages.CopyFrom(self._internal_messages)
    self._internal_messages.Clear()
    self._respond_result(result)