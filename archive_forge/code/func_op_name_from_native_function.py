from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
def op_name_from_native_function(f: NativeFunction) -> str:
    return f'{f.namespace}::{f.func.name}'