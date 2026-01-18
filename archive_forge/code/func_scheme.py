import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
@property
def scheme(self) -> Partitioning:
    """Returns the partitioning for this parser."""
    return self._scheme