from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@lazyproperty
def parent_pid(self) -> Optional[int]:
    """
        Parent PID
        """
    return self.parent.pid if self.parent else None