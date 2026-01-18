from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@property
def is_leader(self) -> bool:
    """
        If the current process is the leader

        detect the first worker pid by numerical order
        """
    return True if self.is_parent else self.leader_pid == self.pid