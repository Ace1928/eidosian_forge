import logging
import os
import pickle
import sys
import threading
import warnings
from types import ModuleType, TracebackType
from typing import Any, Dict, List, Optional, Tuple, Type
from packaging.version import Version
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.enums import LightningEnum
from lightning_fabric.utilities.imports import _IS_WINDOWS
from lightning_fabric.utilities.types import _PATH
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.utilities.migration.migration import _migration_index
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def migrate_checkpoint(checkpoint: _CHECKPOINT, target_version: Optional[str]=None) -> Tuple[_CHECKPOINT, Dict[str, List[str]]]:
    """Applies Lightning version migrations to a checkpoint dictionary.

    Args:
        checkpoint: A dictionary with the loaded state from the checkpoint file.
        target_version: Run migrations only up to this version (inclusive), even if migration index contains
            migration functions for newer versions than this target. Mainly useful for testing.

    Note:
        The migration happens in-place. We specifically avoid copying the dict to avoid memory spikes for large
        checkpoints and objects that do not support being deep-copied.

    """
    ckpt_version = _get_version(checkpoint)
    if Version(ckpt_version) > Version(pl.__version__):
        rank_zero_warn(f'The loaded checkpoint was produced with Lightning v{ckpt_version}, which is newer than your current Lightning version: v{pl.__version__}', category=PossibleUserWarning)
        return (checkpoint, {})
    index = _migration_index()
    applied_migrations = {}
    for migration_version, migration_functions in index.items():
        if not _should_upgrade(checkpoint, migration_version, target_version):
            continue
        for migration_function in migration_functions:
            checkpoint = migration_function(checkpoint)
        applied_migrations[migration_version] = [fn.__name__ for fn in migration_functions]
    if ckpt_version != pl.__version__:
        _set_legacy_version(checkpoint, ckpt_version)
    _set_version(checkpoint, pl.__version__)
    return (checkpoint, applied_migrations)