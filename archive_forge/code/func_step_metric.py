import logging
from typing import Callable, Optional, Sequence, Tuple
from wandb.proto import wandb_internal_pb2 as pb
@property
def step_metric(self) -> Optional[str]:
    return self._step_metric