import logging
import threading
from typing import Dict, Optional
from wandb.proto.wandb_internal_pb2 import Record, Result
@property
def cancel_event(self) -> threading.Event:
    return self._cancel_event