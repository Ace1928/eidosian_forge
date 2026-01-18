import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def deliver(self, result: pb.Result) -> None:
    mailbox = result.control.mailbox_slot
    slot = self._slots.get(mailbox)
    if not slot:
        return
    slot._deliver(result)