import asyncio
import sys
from copy import deepcopy
from collections import defaultdict
import concurrent.futures
from dataclasses import dataclass, field
import logging
import numpy as np
import pprint
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple, Union
from ray.util.state import list_tasks
import ray
from ray.actor import ActorHandle
from ray.util.state import list_workers
from ray._private.gcs_utils import GcsAioClient, GcsChannel
from ray.util.state.state_manager import StateDataSourceClient
from ray.dashboard.state_aggregator import (
def periodic_invoke_state_apis_with_actor(*args, **kwargs) -> ActorHandle:
    current_node_ip = ray._private.worker.global_worker.node_ip_address
    actor = StateAPIGeneratorActor.options(resources={f'node:{current_node_ip}': 0.001}).remote(*args, **kwargs)
    print('Waiting for state api actor to be ready...')
    ray.get(actor.ready.remote())
    print('State api actor is ready now.')
    actor.start.remote()
    return actor