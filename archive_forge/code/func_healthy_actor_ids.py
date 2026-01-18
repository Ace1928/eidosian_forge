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
def healthy_actor_ids(self) -> List[int]:
    """Returns a list of worker IDs that are healthy."""
    return [k for k, v in self.__remote_actor_states.items() if v.is_healthy]