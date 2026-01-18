import logging
import math
import time
import warnings
from collections import defaultdict
from typing import Dict, List
import numpy as np
import pandas
import ray
import xgboost as xgb
from ray.util import get_node_ip_address
from modin.core.execution.ray.common import RayWrapper
from modin.distributed.dataframe.pandas import from_partitions
from .utils import RabitContext, RabitContextManager
def create_actors(num_actors):
    """
    Create ModinXGBoostActors.

    Parameters
    ----------
    num_actors : int
        Number of actors to create.

    Returns
    -------
    list
        List of pairs (ip, actor).
    """
    num_cpus_per_actor = _get_cpus_per_actor(num_actors)
    node_ips = [key for key in ray.cluster_resources().keys() if key.startswith('node:') and '__internal_head__' not in key]
    num_actors_per_node = max(num_actors // len(node_ips), 1)
    actors_ips = [ip for ip in node_ips for _ in range(num_actors_per_node)]
    actors = [(node_ip.split('node:')[-1], ModinXGBoostActor.options(resources={node_ip: 0.01}).remote(i, nthread=num_cpus_per_actor)) for i, node_ip in enumerate(actors_ips)]
    return actors