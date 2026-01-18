import abc
from dataclasses import dataclass
from typing import List, Any
from torch.futures import Future
from .metadata import (
from .planner import (
@abc.abstractmethod
def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
    """
        Initialize this instance.

        Args:
            metadata (Metadata): The metadata schema to use.
            is_coordinator (bool): Whether this instance is responsible for coordinating
              the checkpoint.
        """
    pass