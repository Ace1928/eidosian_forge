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
def handle_summary(self, record: Record) -> None:
    summary = record.summary
    for item in summary.update:
        if len(item.nested_key) > 0:
            assert item.key == ''
            key = tuple(item.nested_key)
        else:
            key = (item.key,)
        target = self._consolidated_summary
        for prop in key[:-1]:
            target = target[prop]
        target[key[-1]] = json.loads(item.value_json)
    for item in summary.remove:
        if len(item.nested_key) > 0:
            assert item.key == ''
            key = tuple(item.nested_key)
        else:
            key = (item.key,)
        target = self._consolidated_summary
        for prop in key[:-1]:
            target = target[prop]
        del target[key[-1]]
    self._save_summary(self._consolidated_summary)