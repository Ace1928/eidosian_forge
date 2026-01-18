import json
import hashlib
import datetime
from typing import Any, Dict, List, Union, Optional
from collections import OrderedDict
from libcloud.compute.base import Node, NodeSize, NodeImage
from libcloud.compute.types import NodeState
from libcloud.container.base import Container, ContainerImage, ContainerDriver, ContainerCluster
from libcloud.container.types import ContainerState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.common.kubernetes import (
from libcloud.container.providers import Provider
def sum_resources(*resource_dicts):
    total_cpu = 0
    total_memory = 0
    for rd in resource_dicts:
        total_cpu += to_n_cpus(rd.get('cpu', '0m'))
        total_memory += to_n_bytes(rd.get('memory', '0K'))
    return {'cpu': to_cpu_str(total_cpu), 'memory': to_memory_str(total_memory)}