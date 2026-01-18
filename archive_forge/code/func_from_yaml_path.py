from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
@staticmethod
def from_yaml_path(config_path: str) -> 'SelectiveBuilder':
    with open(config_path) as f:
        contents = yaml.safe_load(f)
        return SelectiveBuilder.from_yaml_dict(contents)