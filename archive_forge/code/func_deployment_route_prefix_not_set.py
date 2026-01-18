import logging
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union
from zlib import crc32
from ray._private.pydantic_compat import (
from ray._private.runtime_env.packaging import parse_uri
from ray.serve._private.common import (
from ray.serve._private.constants import (
from ray.serve._private.deployment_info import DeploymentInfo
from ray.serve._private.utils import DEFAULT
from ray.serve.config import ProxyLocation
from ray.util.annotations import PublicAPI
@validator('deployment_config')
def deployment_route_prefix_not_set(cls, v: DeploymentSchema):
    if 'route_prefix' in v.dict(exclude_unset=True):
        raise ValueError(f'Unexpectedly found a deployment-level route_prefix in the deployment_config for deployment "{cls.name}". The route_prefix in deployment_config within DeploymentDetails should not be set; please set it at the application level.')
    return v