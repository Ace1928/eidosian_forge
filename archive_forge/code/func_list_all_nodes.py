import copy
import json
import logging
import os
import subprocess
import tempfile
import time
from typing import Dict, Optional
import yaml
import ray
import ray._private.services
from ray._private import ray_constants
from ray._private.client_mode_hook import disable_client_hook
from ray._raylet import GcsClientOptions
from ray.util.annotations import DeveloperAPI
def list_all_nodes(self):
    """Lists all nodes.

        TODO(rliaw): What is the desired behavior if a head node
        dies before worker nodes die?

        Returns:
            List of all nodes, including the head node.
        """
    nodes = list(self.worker_nodes)
    if self.head_node:
        nodes = [self.head_node] + nodes
    return nodes