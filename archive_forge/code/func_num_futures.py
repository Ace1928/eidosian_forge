import random
import ray
from typing import Any, Callable, Dict, Iterable, Optional, Set, Tuple, Union
@property
def num_futures(self) -> int:
    return len(self._tracked_futures)