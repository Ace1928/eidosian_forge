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
@_retry_on_exception(HttpError, 'unable to queue the operation')
def set_labels(self, node: GCPTPUNode, labels: dict, wait_for_operation: bool=True) -> dict:
    body = {'labels': dict(node['labels'], **labels)}
    update_mask = 'labels'
    operation = self.resource.projects().locations().nodes().patch(name=node['name'], updateMask=update_mask, body=body).execute()
    if wait_for_operation:
        result = self.wait_for_operation(operation)
    else:
        result = operation
    return result