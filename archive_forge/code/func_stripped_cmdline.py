from lazyops.imports._psutil import resolve_psutil
import os
import psutil
from typing import List, Optional
from lazyops.types import BaseModel, lazyproperty
@lazyproperty
def stripped_cmdline(self) -> List[str]:
    """
        Strips the current process from the cmdline
        and removes params that are not needed
        """
    return self.get_stripped_cmdline(self.cmdline)