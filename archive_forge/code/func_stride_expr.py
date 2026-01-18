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
def stride_expr(name: str) -> str:
    assert var_names == (name,), 'Replacement for ".strides()" is currently only supported for single derivatives of the same tensor that ".strides()" is being called on.'
    return f'strides_or_error({name}, "{name}")'