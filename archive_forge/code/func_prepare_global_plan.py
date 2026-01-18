import abc
from dataclasses import dataclass
from typing import List, Any
from torch.futures import Future
from .metadata import (
from .planner import (
@abc.abstractmethod
def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
    """
        Perform centralized planning of storage loading.

        This method is only called on the coordinator instance.

        While this method can produce a completely different plan, the preferred
        way is to store storage specific data in LoadPlan::storage_data.

        Args:
            plans: A list of ``LoadPlan`` instances, one for each rank.

        Returns:
            A list of transformed ``LoadPlan`` after storage global planning
        """
    pass