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
def schedule_actor_task(self, tracked_actor: TrackedActor, method_name: str, args: Optional[Tuple]=None, kwargs: Optional[Dict]=None, on_result: Optional[Callable[[TrackedActor, Any], None]]=None, on_error: Optional[Callable[[TrackedActor, Exception], None]]=None, _return_future: bool=False) -> Optional[ray.ObjectRef]:
    """Schedule and track a task on an actor.

        This method will schedule a remote task ``method_name`` on the
        ``tracked_actor``.

        This method accepts two optional callbacks that will be invoked when
        their respective events are triggered.

        The ``on_result`` callback is triggered when a task resolves successfully.
        It should accept two arguments: The actor for which the
        task resolved, and the result received from the remote call.

        The ``on_error`` callback is triggered when a task fails.
        It should accept two arguments: The actor for which the
        task threw an error, and the exception.

        Args:
            tracked_actor: Actor to schedule task on.
            method_name: Remote method name to invoke on the actor. If this is
                e.g. ``foo``, then ``actor.foo.remote(*args, **kwargs)`` will be
                scheduled.
            args: Arguments to pass to the task.
            kwargs: Keyword arguments to pass to the task.
            on_result: Callback to invoke when the task resolves.
            on_error: Callback to invoke when the task fails.

        Raises:
            ValueError: If the ``tracked_actor`` is not managed by this event manager.

        """
    args = args or tuple()
    kwargs = kwargs or {}
    if tracked_actor.actor_id in self._failed_actor_ids:
        return
    tracked_actor_task = TrackedActorTask(tracked_actor=tracked_actor, on_result=on_result, on_error=on_error)
    if tracked_actor not in self._live_actors_to_ray_actors_resources:
        if tracked_actor not in self._pending_actors_to_attrs:
            raise ValueError(f'Tracked actor is not managed by this event manager: {tracked_actor}')
        self._pending_actors_to_enqueued_actor_tasks[tracked_actor].append((tracked_actor_task, method_name, args, kwargs))
    else:
        res = self._schedule_tracked_actor_task(tracked_actor_task=tracked_actor_task, method_name=method_name, args=args, kwargs=kwargs, _return_future=_return_future)
        if _return_future:
            return res[1]