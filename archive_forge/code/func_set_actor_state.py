from collections import defaultdict
import copy
from dataclasses import dataclass
import logging
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple, Union
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError, RayError, RayTaskError
from ray.rllib.utils.typing import T
from ray.util.annotations import DeveloperAPI
@DeveloperAPI
def set_actor_state(self, actor_id: int, healthy: bool) -> None:
    """Update activate state for a specific remote actor.

        Args:
            actor_id: ID of the remote actor.
            healthy: Whether the remote actor is healthy.
        """
    if actor_id not in self.__remote_actor_states:
        raise ValueError(f'Unknown actor id: {actor_id}')
    self.__remote_actor_states[actor_id].is_healthy = healthy
    if not healthy:
        self._remove_async_state(actor_id)