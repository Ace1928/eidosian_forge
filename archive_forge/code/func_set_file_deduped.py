import threading
from typing import MutableMapping, NamedTuple
import wandb
def set_file_deduped(self, save_name: str) -> None:
    with self._lock:
        orig = self._stats[save_name]
        self._stats[save_name] = orig._replace(deduped=True, uploaded=orig.total)