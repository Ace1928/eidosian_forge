from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@property
def leader_pid(self) -> Optional[int]:
    """
        Returns the leader pid
        """
    if self.is_worker:
        return self.sorted_linked_pids[1] if len(self.sorted_linked_pids) > 1 else None
    return self.parent_pid if self.is_parent else self.sorted_linked_pids[0]