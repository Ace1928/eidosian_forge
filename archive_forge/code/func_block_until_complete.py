import threading
from typing import Any, List, Optional
import ray
from ray.experimental import tqdm_ray
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def block_until_complete(self, remaining: List[ObjectRef]) -> None:
    t = threading.current_thread()
    while remaining:
        done, remaining = ray.wait(remaining, fetch_local=False, timeout=0.1)
        self.update(len(done))
        with _canceled_threads_lock:
            if t in _canceled_threads:
                break