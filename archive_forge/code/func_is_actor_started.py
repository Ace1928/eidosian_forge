import logging
import random
import time
import uuid
from collections import defaultdict, Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union
import ray
from ray.air.execution._internal.event_manager import RayEventManager
from ray.air.execution.resources import (
from ray.air.execution._internal.tracked_actor import TrackedActor
from ray.air.execution._internal.tracked_actor_task import TrackedActorTask
from ray.exceptions import RayTaskError, RayActorError
def is_actor_started(self, tracked_actor: TrackedActor) -> bool:
    """Returns True if the actor has been started.

        Args:
            tracked_actor: Tracked actor object.
        """
    return tracked_actor in self._live_actors_to_ray_actors_resources and tracked_actor.actor_id not in self._failed_actor_ids