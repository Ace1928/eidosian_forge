from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
def is_native_function_selected_for_training(self, func: NativeFunction) -> bool:
    op_name = op_name_from_native_function(func)
    return self.is_operator_selected_for_training(op_name)