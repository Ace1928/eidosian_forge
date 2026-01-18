from ray.util.client import ray
from typing import Tuple
@ray.remote
def get_nodes():
    return ray.nodes()