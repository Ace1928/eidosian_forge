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
def list_instances(self, label_filters: Optional[dict]=None, is_terminated: bool=False) -> List[GCPTPUNode]:
    response = self.resource.projects().locations().nodes().list(parent=self.path).execute()
    instances = response.get('nodes', [])
    instances = [GCPTPUNode(i, self) for i in instances]
    label_filters = label_filters or {}
    label_filters[TAG_RAY_CLUSTER_NAME] = self.cluster_name

    def filter_instance(instance: GCPTPUNode) -> bool:
        if instance.is_terminated():
            return False
        labels = instance.get_labels()
        if label_filters:
            for key, value in label_filters.items():
                if key not in labels:
                    return False
                if value != labels[key]:
                    return False
        return True
    instances = list(filter(filter_instance, instances))
    return instances