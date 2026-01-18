import atexit
import logging
import os
import queue
import sys
import threading
import time
import traceback
from datetime import datetime
from typing import TYPE_CHECKING, Any, List, Optional
import psutil
import wandb
from ..interface.interface_queue import InterfaceQueue
from ..lib import tracelog
from . import context, handler, internal_util, sender, writer
def close_internal_log() -> None:
    root = logging.getLogger('wandb')
    for _handler in root.handlers[:]:
        _handler.close()
        root.removeHandler(_handler)