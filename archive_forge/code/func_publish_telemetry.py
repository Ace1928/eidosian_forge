import datetime
import logging
import queue
import threading
from collections import defaultdict, deque
from typing import TYPE_CHECKING, Deque, Dict, List, Optional, Tuple
from .assets.asset_registry import asset_registry
from .assets.interfaces import Asset, Interface
from .assets.open_metrics import OpenMetrics
from .system_info import SystemInfo
def publish_telemetry(self) -> None:
    if self.asset_interface is None:
        return None
    while not self.asset_interface.telemetry_queue.empty():
        telemetry_record = self.asset_interface.telemetry_queue.get()
        self.backend_interface._publish_telemetry(telemetry_record)