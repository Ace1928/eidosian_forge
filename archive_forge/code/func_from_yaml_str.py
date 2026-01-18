from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
@staticmethod
def from_yaml_str(config_contents: str) -> 'SelectiveBuilder':
    contents = yaml.safe_load(config_contents)
    return SelectiveBuilder.from_yaml_dict(contents)