import logging
from typing import Callable, Optional, Sequence, Tuple
from wandb.proto import wandb_internal_pb2 as pb
@property
def hidden(self) -> Optional[bool]:
    return self._hidden