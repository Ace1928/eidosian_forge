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
def remove_actor(self, tracked_actor: TrackedActor, kill: bool=False, stop_future: Optional[ray.ObjectRef]=None) -> bool:
    """Remove a tracked actor.

        If the actor has already been started, this will stop the actor. This will
        trigger the :meth:`TrackedActor.on_stop
        <ray.air.execution._internal.tracked_actor.TrackedActor.on_stop>`
        callback once the actor stopped.

        If the actor has only been requested, but not started, yet, this will cancel
        the actor request. This will not trigger any callback.

        If ``kill=True``, this will use ``ray.kill()`` to forcefully terminate the
        actor. Otherwise, graceful actor deconstruction will be scheduled after
        all currently tracked futures are resolved.

        This method returns a boolean, indicating if a stop future is tracked and
        the ``on_stop`` callback will be invoked. If the actor has been alive,
        this will be ``True``. If the actor hasn't been scheduled, yet, or failed
        (and triggered the ``on_error`` callback), this will be ``False``.

        Args:
            tracked_actor: Tracked actor to be removed.
            kill: If set, will forcefully terminate the actor instead of gracefully
                scheduling termination.
            stop_future: If set, use this future to track actor termination.
                Otherwise, schedule a ``__ray_terminate__`` future.

        Returns:
            Boolean indicating if the actor was previously alive, and thus whether
            a callback will be invoked once it is terminated.

        """
    if tracked_actor.actor_id in self._failed_actor_ids:
        logger.debug(f'Tracked actor already failed, no need to remove: {tracked_actor}')
        return False
    elif tracked_actor in self._live_actors_to_ray_actors_resources:
        if not kill:
            ray_actor, _ = self._live_actors_to_ray_actors_resources[tracked_actor]
            for future in list(self._tracked_actors_to_state_futures[tracked_actor]):
                self._actor_state_events.discard_future(future)
                self._tracked_actors_to_state_futures[tracked_actor].remove(future)
                tracked_actor._on_start = None
                tracked_actor._on_stop = None
                tracked_actor._on_error = None

            def on_actor_stop(*args, **kwargs):
                self._actor_stop_resolved(tracked_actor=tracked_actor)
            if stop_future:
                self._actor_task_events.discard_future(stop_future)
            else:
                stop_future = ray_actor.__ray_terminate__.remote()
            self._actor_state_events.track_future(future=stop_future, on_result=on_actor_stop, on_error=on_actor_stop)
            self._tracked_actors_to_state_futures[tracked_actor].add(stop_future)
        else:
            self._live_actors_to_kill.add(tracked_actor)
        return True
    elif tracked_actor in self._pending_actors_to_attrs:
        _, _, resource_request = self._pending_actors_to_attrs.pop(tracked_actor)
        self._resource_request_to_pending_actors[resource_request].remove(tracked_actor)
        self._resource_manager.cancel_resource_request(resource_request=resource_request)
        return False
    else:
        raise ValueError(f'Unknown tracked actor: {tracked_actor}')