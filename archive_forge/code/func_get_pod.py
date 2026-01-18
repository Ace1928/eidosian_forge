from __future__ import annotations
import random
import asyncio
import logging
import contextlib
from enum import Enum
from lazyops.libs.kops.base import *
from lazyops.libs.kops.config import KOpsSettings
from lazyops.libs.kops.utils import cached, DillSerializer, SignalHandler
from lazyops.libs.kops._kopf import kopf
from lazyops.types import lazyproperty
from lazyops.utils import logger
from typing import List, Dict, Union, Any, Optional, Callable, TYPE_CHECKING
import lazyops.libs.kops.types as t
import lazyops.libs.kops.atypes as at
def get_pod(name: str, namespace: Optional[str]=None, **kwargs) -> t.V1Pod:
    if name and namespace:
        return BaseKOpsClient.core_v1.read_namespaced_pod(name=name, namespace=namespace, **kwargs)
    pods = get_pods(namespace=namespace, **kwargs)
    for pod in pods:
        if pod.metadata.name == name:
            return pod
    raise ValueError(f'Pod {name} not found')