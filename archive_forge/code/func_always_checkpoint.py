from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
from triad import SerializableRLock, to_uuid
from tune.concepts.flow import (
@property
def always_checkpoint(self) -> bool:
    return self._always_checkpoint