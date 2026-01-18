from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
def is_operator_selected(self, name: str) -> bool:
    if self.include_all_operators:
        return True
    if name in self.operators:
        return True
    name = strip_operator_overload_name(name)
    return name in self.operators and self.operators[name].include_all_overloads