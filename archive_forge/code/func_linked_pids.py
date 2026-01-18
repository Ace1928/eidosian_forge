from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@lazyproperty
def linked_pids(self) -> Optional[List[int]]:
    """
        Linked PID's
        """
    return [p.pid for p in self.linked] if self.linked else None