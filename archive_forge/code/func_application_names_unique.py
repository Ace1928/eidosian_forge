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
def application_names_unique(cls, v):
    names = [app.name for app in v]
    duplicates = {f'"{name}"' for name in names if names.count(name) > 1}
    if len(duplicates):
        apps_str = ('application ' if len(duplicates) == 1 else 'applications ') + ', '.join(duplicates)
        raise ValueError(f'Found multiple configs for {apps_str}. Please remove all duplicates.')
    return v