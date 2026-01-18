from __future__ import annotations
import os
import pickle
import typing as t
from .constants import (
from .compat.packaging import (
from .compat.yaml import (
from .io import (
from .util import (
from .data import (
from .config import (
def serialize_content_config(args: EnvironmentConfig, path: str) -> None:
    """Serialize the content config to the given path. If the config has not been loaded, an empty config will be serialized."""
    with open_binary_file(path, 'wb') as config_file:
        pickle.dump(args.content_config, config_file)