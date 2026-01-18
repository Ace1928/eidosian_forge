import abc
import json
from copy import deepcopy
from inspect import signature
from typing import Dict, List, Union
from dataclasses import dataclass
import ray
from ray.util import placement_group
from ray.util.annotations import DeveloperAPI
@property
def required_resources(self) -> Dict[str, float]:
    """Returns a dict containing the sums of all resources"""
    return _sum_bundles(self._bundles)