import threading
from typing import Any, List, Optional
import ray
from ray.experimental import tqdm_ray
from ray.types import ObjectRef
from ray.util.annotations import PublicAPI
def set_description(self, name: str) -> None:
    if self._bar and name != self._desc:
        self._desc = name
        self._bar.set_description(self._desc)