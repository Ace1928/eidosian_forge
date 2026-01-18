import importlib.resources as pkg_resources
import logging
from pathlib import Path
from typing import Any, List, Tuple
import yaml
from . import resources
from .deprecation_utils import deprecated
def load_yaml_resource(resource: str) -> Tuple[Any, str]:
    content = pkg_resources.read_text(resources, resource)
    return (yaml.safe_load(content), f'{BASE_REF_URL}/resources/{resource}')