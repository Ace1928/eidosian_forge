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
def schedule_actor_tasks(self, tracked_actors: List[TrackedActor], method_name: str, *, args: Optional[Union[Tuple, List[Tuple]]]=None, kwargs: Optional[Union[Dict, List[Dict]]]=None, on_result: Optional[Callable[[TrackedActor, Any], None]]=None, on_error: Optional[Callable[[TrackedActor, Exception], None]]=None) -> None:
    """Schedule and track tasks on a list of actors.

        This method will schedule a remote task ``method_name`` on all
        ``tracked_actors``.

        ``args`` and ``kwargs`` can be a single tuple/dict, in which case the same
        (keyword) arguments are passed to all actors. If a list is passed instead,
        they are mapped to the respective actors. In that case, the list of
        (keyword) arguments must be the same length as the list of actors.

        This method accepts two optional callbacks that will be invoked when
        their respective events are triggered.

        The ``on_result`` callback is triggered when a task resolves successfully.
        It should accept two arguments: The actor for which the
        task resolved, and the result received from the remote call.

        The ``on_error`` callback is triggered when a task fails.
        It should accept two arguments: The actor for which the
        task threw an error, and the exception.

        Args:
            tracked_actors: List of actors to schedule tasks on.
            method_name: Remote actor method to invoke on the actors. If this is
                e.g. ``foo``, then ``actor.foo.remote(*args, **kwargs)`` will be
                scheduled on all actors.
            args: Arguments to pass to the task.
            kwargs: Keyword arguments to pass to the task.
            on_result: Callback to invoke when the task resolves.
            on_error: Callback to invoke when the task fails.

        """
    if not isinstance(args, List):
        args_list = [args] * len(tracked_actors)
    else:
        if len(tracked_actors) != len(args):
            raise ValueError(f'Length of args must be the same as tracked_actors list. Got `len(kwargs)={len(kwargs)}` and `len(tracked_actors)={len(tracked_actors)}')
        args_list = args
    if not isinstance(kwargs, List):
        kwargs_list = [kwargs] * len(tracked_actors)
    else:
        if len(tracked_actors) != len(kwargs):
            raise ValueError(f'Length of kwargs must be the same as tracked_actors list. Got `len(args)={len(args)}` and `len(tracked_actors)={len(tracked_actors)}')
        kwargs_list = kwargs
    for tracked_actor, args, kwargs in zip(tracked_actors, args_list, kwargs_list):
        self.schedule_actor_task(tracked_actor=tracked_actor, method_name=method_name, args=args, kwargs=kwargs, on_result=on_result, on_error=on_error)