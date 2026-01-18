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
def foreach_actor_async(self, func: Union[Callable[[Any], Any], List[Callable[[Any], Any]]], tag: str=None, *, healthy_only=True, remote_actor_ids: List[int]=None) -> int:
    """Calls given functions against each actors without waiting for results.

        Args:
            func: A single, or a list of Callables, that get applied on the list
                of specified remote actors.
            tag: A tag to identify the results from this async call.
            healthy_only: If True, applies func on known healthy actors only.
            remote_actor_ids: Apply func on a selected set of remote actors.
                Note, for fault tolerance reasons, these returned ObjectRefs should
                never be resolved with ray.get() outside of the context of this manager.

        Returns:
            The number of async requests that are actually fired.
        """
    remote_actor_ids = remote_actor_ids or self.actor_ids()
    if healthy_only:
        func, remote_actor_ids = self._filter_func_and_remote_actor_id_by_state(func, remote_actor_ids)
    if isinstance(func, list) and len(func) != len(remote_actor_ids):
        raise ValueError(f'The number of functions specified {len(func)} must match the number of remote actor indices {len(remote_actor_ids)}.')
    num_calls_to_make: Dict[int, int] = defaultdict(lambda: 0)
    if isinstance(func, list):
        limited_func = []
        limited_remote_actor_ids = []
        for i, f in zip(remote_actor_ids, func):
            num_outstanding_reqs = self.__remote_actor_states[i].num_in_flight_async_requests
            if num_outstanding_reqs + num_calls_to_make[i] < self._max_remote_requests_in_flight_per_actor:
                num_calls_to_make[i] += 1
                limited_func.append(f)
                limited_remote_actor_ids.append(i)
    else:
        limited_func = func
        limited_remote_actor_ids = []
        for i in remote_actor_ids:
            num_outstanding_reqs = self.__remote_actor_states[i].num_in_flight_async_requests
            if num_outstanding_reqs + num_calls_to_make[i] < self._max_remote_requests_in_flight_per_actor:
                num_calls_to_make[i] += 1
                limited_remote_actor_ids.append(i)
    remote_calls = self.__call_actors(func=limited_func, remote_actor_ids=limited_remote_actor_ids)
    for id, call in zip(limited_remote_actor_ids, remote_calls):
        self.__remote_actor_states[id].num_in_flight_async_requests += 1
        self.__in_flight_req_to_actor_id[call] = (tag, id)
    return len(remote_calls)