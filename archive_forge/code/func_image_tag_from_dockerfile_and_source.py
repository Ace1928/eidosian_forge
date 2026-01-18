import hashlib
import json
import logging
import os
import pathlib
import shlex
import shutil
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple
import yaml
from dockerpycreds.utils import find_executable  # type: ignore
from six.moves import shlex_quote
import wandb
import wandb.docker as docker
import wandb.env
from wandb.apis.internal import Api
from wandb.sdk.launch.loader import (
from wandb.util import get_module
from .._project_spec import EntryPoint, EntrypointDefaults, LaunchProject
from ..errors import ExecutionError, LaunchError
from ..registry.abstract import AbstractRegistry
from ..registry.anon import AnonynmousRegistry
from ..utils import (
def image_tag_from_dockerfile_and_source(launch_project: LaunchProject, dockerfile_contents: str) -> str:
    """Hashes the source and dockerfile contents into a unique tag."""
    image_source_string = launch_project.get_image_source_string()
    unique_id_string = image_source_string + dockerfile_contents
    image_tag = hashlib.sha256(unique_id_string.encode('utf-8')).hexdigest()[:8]
    return image_tag