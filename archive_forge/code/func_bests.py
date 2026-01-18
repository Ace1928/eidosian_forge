from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
@property
def bests(self) -> List[float]:
    with self._lock:
        return self._bests