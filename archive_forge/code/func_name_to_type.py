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
@staticmethod
def name_to_type(name: str):
    """Provided a node name, determine the type.

        This expects the name to be in format '[NAME]-[UUID]-[TYPE]',
        where [TYPE] is either 'compute' or 'tpu'.
        """
    return GCPNodeType(name.split('-')[-1])