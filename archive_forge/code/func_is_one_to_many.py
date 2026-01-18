from __future__ import annotations
import abc
from typing import TYPE_CHECKING
from monty.json import MSONable
@property
@abc.abstractmethod
def is_one_to_many(self) -> bool:
    """Determines if a Transformation is a one-to-many transformation. If a
        Transformation is a one-to-many transformation, the
        apply_transformation method should have a keyword arg
        "return_ranked_list" which allows for the transformed structures to be
        returned as a ranked list.
        """
    return False