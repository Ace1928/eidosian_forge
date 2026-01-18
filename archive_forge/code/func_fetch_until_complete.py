import threading
from typing import Any, List, Optional
import ray
from ray.experimental import tqdm_ray
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def fetch_until_complete(self, refs: List[ObjectRef]) -> List[Any]:
    ref_to_result = {}
    remaining = refs
    t = threading.current_thread()
    fetch_local = True
    while remaining:
        done, remaining = ray.wait(remaining, fetch_local=fetch_local, timeout=0.1)
        if fetch_local:
            fetch_local = False
        for ref, result in zip(done, ray.get(done)):
            ref_to_result[ref] = result
        self.update(len(done))
        with _canceled_threads_lock:
            if t in _canceled_threads:
                break
    return [ref_to_result[ref] for ref in refs]