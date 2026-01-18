import os
from dataclasses import dataclass, field
from typing import List, Optional, Union
from .common import NodeIdStr
from ray.data._internal.execution.util import memory_string
from ray.util.annotations import DeveloperAPI
def satisfies_limit(self, limit: 'ExecutionResources') -> bool:
    """Return if this resource struct meets the specified limits.

        Note that None for a field means no limit.
        """
    if self.cpu is not None and limit.cpu is not None and (self.cpu > limit.cpu):
        return False
    if self.gpu is not None and limit.gpu is not None and (self.gpu > limit.gpu):
        return False
    if self.object_store_memory is not None and limit.object_store_memory is not None and (self.object_store_memory > limit.object_store_memory):
        return False
    return True