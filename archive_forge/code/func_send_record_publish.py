import socket
import struct
import threading
import time
import uuid
from typing import TYPE_CHECKING, Any, List, Optional
from wandb.proto import wandb_server_pb2 as spb
from . import tracelog
def send_record_publish(self, record: 'pb.Record') -> None:
    server_req = spb.ServerRequest()
    server_req.record_publish.CopyFrom(record)
    self.send_server_request(server_req)