import weakref
from dataclasses import dataclass
import logging
from typing import List, TypeVar, Optional, Dict, Type, Tuple
import ray
from ray.actor import ActorHandle
from ray.util.annotations import Deprecated
from ray._private.utils import get_ray_doc_version
def remove_actors(self, actor_indexes: List[int]):
    """Removes the actors with the specified indexes.

        Args:
            actor_indexes (List[int]): The indexes of the actors to remove.
        """
    new_actors = []
    for i in range(len(self.actors)):
        if i not in actor_indexes:
            new_actors.append(self.actors[i])
    self.actors = new_actors