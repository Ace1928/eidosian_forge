from abc import ABCMeta
import glob
import os
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
import warnings
from ray.util.annotations import PublicAPI, DeveloperAPI
from ray.tune.utils.util import _atomic_save, _load_newest_checkpoint
def can_restore(self, checkpoint_dir: str) -> bool:
    """Check if the checkpoint_dir contains the saved state for this callback list.

        Returns:
            can_restore: True if the checkpoint_dir contains a file of the
                format `CKPT_FILE_TMPL`. False otherwise.
        """
    return bool(glob.glob(os.path.join(checkpoint_dir, self.CKPT_FILE_TMPL.format('*'))))