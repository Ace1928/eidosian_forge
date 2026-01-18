from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml
from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader, parse_native_yaml
from torchgen.model import (
from torchgen.utils import NamespaceHelper
def strip_et_fields(es: object) -> None:
    """Given a loaded yaml representing a list of operators,
    remove ET specific fields from every entries for BC compatibility
    """
    for entry in es:
        for field in ET_FIELDS:
            entry.pop(field, None)