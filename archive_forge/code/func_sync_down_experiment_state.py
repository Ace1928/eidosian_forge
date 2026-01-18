from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple, Union
import click
import logging
import os
import time
import warnings
from ray.train._internal.storage import (
from ray.tune.experiment import Trial
from ray.tune.impl.out_of_band_serialize_dataset import out_of_band_serialize_dataset
def sync_down_experiment_state(self) -> None:
    fs = self._storage.storage_filesystem
    filepaths = _list_at_fs_path(fs=fs, fs_path=self._storage.experiment_fs_path)
    matches = [path for path in filepaths if path.endswith('.json') or path.endswith('.pkl')]
    for relpath in matches:
        fs_path = Path(self._storage.experiment_fs_path, relpath).as_posix()
        local_path = Path(self._storage.experiment_local_path, relpath).as_posix()
        _download_from_fs_path(fs=fs, fs_path=fs_path, local_path=local_path)
    logger.debug(f'Copied {matches} from:\n(fs, path) = ({self._storage.storage_filesystem.type_name}, {self._storage.experiment_fs_path})\n-> {self._storage.experiment_local_path}')