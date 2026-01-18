import threading
from typing import MutableMapping, NamedTuple
import wandb
def init_file(self, save_name: str, size: int, is_artifact_file: bool=False) -> None:
    with self._lock:
        self._stats[save_name] = FileStats(deduped=False, total=size, uploaded=0, failed=False, artifact_file=is_artifact_file)