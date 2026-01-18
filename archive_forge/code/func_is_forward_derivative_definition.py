import re
from collections import defaultdict
from typing import Any, Counter, Dict, List, Match, Optional, Sequence, Set, Tuple
import yaml
from torchgen.api import cpp
from torchgen.api.autograd import (
from torchgen.api.types import (
from torchgen.context import with_native_function
from torchgen.gen import get_grouped_by_view_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.utils import concatMap, IDENT_REGEX, split_name_params
from torchgen.yaml_utils import YamlLoader
def is_forward_derivative_definition(all_arg_names: List[str], names: Tuple[str, ...]) -> bool:
    for name in names:
        if name not in all_arg_names:
            return True
        else:
            return False
    raise RuntimeError('Expected `names` to be non-empty')