import json
import logging
import os
import random
import traceback
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple, Type
import ray
from ray.actor import ActorHandle
from ray.exceptions import RayActorError
from ray.serve._private.cluster_node_info_cache import ClusterNodeInfoCache
from ray.serve._private.common import NodeId, ProxyStatus
from ray.serve._private.constants import (
from ray.serve._private.proxy import ProxyActor
from ray.serve._private.utils import Timer, TimerBase, format_actor_name
from ray.serve.config import DeploymentMode, HTTPOptions, gRPCOptions
from ray.serve.schema import LoggingConfig, ProxyDetails
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
def is_drained(self) -> ProxyWrapperCallStatus:
    """Return whether the proxy actor is drained or not.

        If the ongoing drained check is finished, and the value can be retrieved,
        reset _is_drained_obj_ref to ensure drained check is finished and return
        FINISHED_SUCCEED status. If the ongoing ready check is not finished,
        return PENDING status.
        """
    finished, _ = ray.wait([self._is_drained_obj_ref], timeout=0)
    if finished:
        self._is_drained_obj_ref = None
        is_drained = ray.get(finished[0])
        if is_drained:
            return ProxyWrapperCallStatus.FINISHED_SUCCEED
        else:
            return ProxyWrapperCallStatus.FINISHED_FAILED
    else:
        return ProxyWrapperCallStatus.PENDING