import contextlib
import logging
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, List, Optional, Sized, Union
import torch
import torch.nn.functional as F
from lightning_utilities.core.imports import package_available
from torch import Tensor
from torch.utils.data import Dataset, DistributedSampler, Sampler
from typing_extensions import override
from lightning_fabric.utilities.cloud_io import _is_local_file_protocol
from lightning_fabric.utilities.data import _num_cpus_available
from lightning_fabric.utilities.rank_zero import rank_zero_info
from lightning_fabric.utilities.types import _PATH, ReduceOp
def is_shared_filesystem(strategy: 'Strategy', path: Optional[_PATH]=None, timeout: int=3) -> bool:
    """Checks whether the filesystem under the given path is shared across all processes.

    This function should only be used in a context where distributed is initialized.

    Args:
        strategy: The strategy being used, either from Fabric (``fabric.strategy``) or from Trainer
            (``trainer.strategy``).
        path: The path to check. Defaults to the current working directory. The user must have permissions to write
            to this path or the parent folder, and the filesystem must be writable.
        timeout: If any of the processes can't list the file created by rank 0 within this many seconds, the
            filesystem is determined to be not shared.

    """
    if path is not None and (not _is_local_file_protocol(path)):
        return True
    path = Path(Path.cwd() if path is None else path).resolve()
    if not hasattr(strategy, 'world_size') or strategy.world_size == 1:
        return True
    rank_zero_path = strategy.broadcast(path)
    if not strategy.reduce_boolean_decision(rank_zero_path == path, all=True):
        return False
    if not strategy.reduce_boolean_decision(path.exists(), all=True):
        raise FileNotFoundError(f'Unable to determine if the path belongs to a shared filesystem. The path does not exist: {path}')
    path = path.parent if path.is_file() else path
    check_file = path / '.lightning_shared_fs_check'
    check_file.unlink(missing_ok=True)
    strategy.barrier()
    if strategy.is_global_zero:
        check_file.touch()
        found = True
    else:
        start = time.perf_counter()
        found = False
        while not found and time.perf_counter() - start < timeout:
            found = check_file.exists()
    strategy.barrier()
    all_found = strategy.reduce_boolean_decision(found, all=True)
    with contextlib.suppress(OSError):
        check_file.unlink()
    return all_found