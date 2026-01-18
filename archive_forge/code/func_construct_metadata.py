import logging
import os
import socket
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union
import ray
from ray.actor import ActorHandle
from ray.air._internal.util import exception_cause, skip_exceptions
from ray.types import ObjectRef
from ray.util.placement_group import PlacementGroup
def construct_metadata() -> WorkerMetadata:
    """Creates metadata for this worker.

    This function is expected to be run on the actor.
    """
    node_id = ray.get_runtime_context().get_node_id()
    node_ip = ray.util.get_node_ip_address()
    hostname = socket.gethostname()
    accelerator_ids = ray.get_runtime_context().get_accelerator_ids()
    pid = os.getpid()
    return WorkerMetadata(node_id=node_id, node_ip=node_ip, hostname=hostname, resource_ids=accelerator_ids, pid=pid)