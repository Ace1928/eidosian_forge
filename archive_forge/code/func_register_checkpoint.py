import logging
import numbers
from typing import Any, Callable, List, Optional, Tuple
from ray._private.dict import flatten_dict
from ray.air._internal.util import is_nan
from ray.air.config import MAX
from ray.train import CheckpointConfig
from ray.train._internal.session import _TrainingResult
from ray.train._internal.storage import _delete_fs_path
def register_checkpoint(self, checkpoint_result: _TrainingResult):
    """Register new checkpoint and add to bookkeeping.

        This method will register a new checkpoint and add it to the internal
        bookkeeping logic. This means the checkpoint manager will decide if
        this checkpoint should be kept, and if older or worse performing
        checkpoints should be deleted.

        Args:
            checkpoint: Tracked checkpoint object to add to bookkeeping.
        """
    self._latest_checkpoint_result = checkpoint_result
    if self._checkpoint_config.checkpoint_score_attribute is not None:
        _insert_into_sorted_list(self._checkpoint_results, checkpoint_result, key=self._get_checkpoint_score)
    else:
        self._checkpoint_results.append(checkpoint_result)
    if self._checkpoint_config.num_to_keep is not None:
        worst_results = set(self._checkpoint_results[:-self._checkpoint_config.num_to_keep])
        results_to_delete = worst_results - {self._latest_checkpoint_result}
        self._checkpoint_results = [checkpoint_result for checkpoint_result in self._checkpoint_results if checkpoint_result not in results_to_delete]
        for checkpoint_result in results_to_delete:
            checkpoint = checkpoint_result.checkpoint
            logger.debug('Deleting checkpoint: ', checkpoint)
            _delete_fs_path(fs=checkpoint.filesystem, fs_path=checkpoint.path)