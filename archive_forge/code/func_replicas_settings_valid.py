import logging
import warnings
from enum import Enum
from typing import Any, Callable, List, Optional, Union
from ray._private.pydantic_compat import (
from ray._private.utils import import_attr
from ray.serve._private.constants import (
from ray.util.annotations import Deprecated, PublicAPI
@validator('max_replicas', always=True)
def replicas_settings_valid(cls, max_replicas, values):
    min_replicas = values.get('min_replicas')
    initial_replicas = values.get('initial_replicas')
    if min_replicas is not None and max_replicas < min_replicas:
        raise ValueError(f'max_replicas ({max_replicas}) must be greater than or equal to min_replicas ({min_replicas})!')
    if initial_replicas is not None:
        if initial_replicas < min_replicas:
            raise ValueError(f'min_replicas ({min_replicas}) must be less than or equal to initial_replicas ({initial_replicas})!')
        elif initial_replicas > max_replicas:
            raise ValueError(f'max_replicas ({max_replicas}) must be greater than or equal to initial_replicas ({initial_replicas})!')
    return max_replicas