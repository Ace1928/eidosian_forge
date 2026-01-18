import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def send_server_request(self, msg: Any) -> None:
    self._send_message(msg)