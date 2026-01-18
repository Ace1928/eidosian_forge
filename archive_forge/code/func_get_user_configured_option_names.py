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
def get_user_configured_option_names(self) -> Set[str]:
    """Get set of names for all user-configured options.

        Any field not set to DEFAULT.VALUE is considered a user-configured option.
        """
    return {field for field, value in self.dict().items() if value is not DEFAULT.VALUE}