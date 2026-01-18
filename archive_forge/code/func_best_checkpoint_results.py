import logging
import numbers
from typing import Any, Callable, List, Optional, Tuple
from ray._private.dict import flatten_dict
from ray.air._internal.util import is_nan
from ray.air.config import MAX
from ray.train import CheckpointConfig
from ray.train._internal.session import _TrainingResult
from ray.train._internal.storage import _delete_fs_path
@property
def best_checkpoint_results(self) -> List[_TrainingResult]:
    if self._checkpoint_config.num_to_keep is None:
        return self._checkpoint_results
    return self._checkpoint_results[-self._checkpoint_config.num_to_keep:]