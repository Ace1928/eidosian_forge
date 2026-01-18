import logging
import threading
from typing import Dict, Optional
from wandb.proto.wandb_internal_pb2 import Record, Result
def context_id_from_result(result: Result) -> str:
    context_id = result.control.mailbox_slot
    return context_id