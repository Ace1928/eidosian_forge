import logging
from typing import Callable, Optional, Sequence, Tuple
from wandb.proto import wandb_internal_pb2 as pb
@property
def goal(self) -> Optional[str]:
    goal_dict = dict(min='minimize', max='maximize')
    return goal_dict[self._goal] if self._goal else None