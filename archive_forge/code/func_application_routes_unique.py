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
@validator('applications')
def application_routes_unique(cls, v):
    routes = [app.route_prefix for app in v if app.route_prefix is not None]
    duplicates = {f'"{route}"' for route in routes if routes.count(route) > 1}
    if len(duplicates):
        routes_str = ('route prefix ' if len(duplicates) == 1 else 'route prefixes ') + ', '.join(duplicates)
        raise ValueError(f"Found duplicate applications for {routes_str}. Please ensure each application's route_prefix is unique.")
    return v