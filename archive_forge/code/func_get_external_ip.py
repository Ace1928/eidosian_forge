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
def get_external_ip(self, worker_index: int=0) -> str:
    return self.get_external_ips()[worker_index].get('accessConfig', {}).get('externalIp', None)