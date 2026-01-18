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
def foreach_actor(self, func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]], *, healthy_only=True, remote_actor_ids: List[int]=None, timeout_seconds=None, return_obj_refs: bool=False, mark_healthy: bool=False) -> RemoteCallResults:
    """Calls the given function with each actor instance as arg.

        Automatically mark actors unhealthy if they fail to respond.

        Args:
            func: A single, or a list of Callables, that get applied on the list
                of specified remote actors.
            healthy_only: If True, applies func on known healthy actors only.
            remote_actor_ids: Apply func on a selected set of remote actors.
            timeout_seconds: Ray.get() timeout. Default is None.
                Note(jungong) : setting timeout_seconds to 0 effectively makes all the
                remote calls fire-and-forget, while setting timeout_seconds to None
                make them synchronous calls.
            return_obj_refs: whether to return ObjectRef instead of actual results.
                Note, for fault tolerance reasons, these returned ObjectRefs should
                never be resolved with ray.get() outside of the context of this manager.
            mark_healthy: whether to mark certain actors healthy based on the results
                of these remote calls. Useful, for example, to make sure actors
                do not come back without proper state restoration.

        Returns:
            The list of return values of all calls to `func(actor)`. The values may be
            actual data returned or exceptions raised during the remote call in the
            format of RemoteCallResults.
        """
    remote_actor_ids = remote_actor_ids or self.actor_ids()
    if healthy_only:
        func, remote_actor_ids = self._filter_func_and_remote_actor_id_by_state(func, remote_actor_ids)
    remote_calls = self.__call_actors(func=func, remote_actor_ids=remote_actor_ids)
    _, remote_results = self.__fetch_result(remote_actor_ids=remote_actor_ids, remote_calls=remote_calls, tags=[None] * len(remote_calls), timeout_seconds=timeout_seconds, return_obj_refs=return_obj_refs, mark_healthy=mark_healthy)
    return remote_results