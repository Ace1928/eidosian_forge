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
def create_callbacks(tracked_actor: TrackedActor, future: ray.ObjectRef):

    def on_actor_start(result: Any):
        self._actor_start_resolved(tracked_actor=tracked_actor, future=future)

    def on_error(exception: Exception):
        self._actor_start_failed(tracked_actor=tracked_actor, exception=exception)
    return (on_actor_start, on_error)