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
def handle_tbrecord(self, record: Record) -> None:
    logger.info('handling tbrecord: %s', record)
    if self._tb_watcher:
        tbrecord = record.tbrecord
        self._tb_watcher.add(tbrecord.log_dir, tbrecord.save, tbrecord.root_dir)
    self._dispatch_record(record)