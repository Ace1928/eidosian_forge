import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def set_mailbox_handle(self, handle: 'MailboxHandle') -> None:
    self._handle = handle