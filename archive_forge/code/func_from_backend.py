import inspect
import logging
import os
import pickle
import socket
import threading
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast
from torch.distributed import PrefixStore, Store
from torch.distributed.elastic.events import (
from .api import (
from .utils import _delay, _PeriodicTimer
@classmethod
def from_backend(cls, run_id: str, store: Store, backend: RendezvousBackend, min_nodes: int, max_nodes: int, local_addr: Optional[str]=None, timeout: Optional[RendezvousTimeout]=None):
    """Create a new :py:class:`DynamicRendezvousHandler`.

        Args:
            run_id:
                The run id of the rendezvous.
            store:
                The C10d store to return as part of the rendezvous.
            backend:
                The backend to use to hold the rendezvous state.
            min_nodes:
                The minimum number of nodes to admit to the rendezvous.
            max_nodes:
                The maximum number of nodes to admit to the rendezvous.
            local_addr:
                The local node address.
            timeout:
                The timeout configuration of the rendezvous.
        """
    node = cls._node_desc_generator.generate(local_addr)
    settings = RendezvousSettings(run_id, min_nodes, max_nodes, timeout or RendezvousTimeout(), keep_alive_interval=timedelta(seconds=5), keep_alive_max_attempt=3)
    state_holder = _BackendRendezvousStateHolder(backend, settings)
    return cls(node, settings, backend.name, store, state_holder)