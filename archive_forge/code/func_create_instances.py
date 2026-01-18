from API responses.
import abc
import logging
import re
import time
from collections import UserDict
from copy import deepcopy
from enum import Enum
from functools import wraps
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4
from googleapiclient.discovery import Resource
from googleapiclient.errors import HttpError
from ray.autoscaler.tags import TAG_RAY_CLUSTER_NAME, TAG_RAY_NODE_NAME
def create_instances(self, base_config: dict, labels: dict, count: int, wait_for_operation: bool=True) -> List[Tuple[dict, str]]:
    """Creates multiple instances and returns result.

        Returns a list of tuples of (result, node_name).
        """
    operations = [self.create_instance(base_config, labels, wait_for_operation=False) for i in range(count)]
    if wait_for_operation:
        results = [(self.wait_for_operation(operation), node_name) for operation, node_name in operations]
    else:
        results = operations
    return results