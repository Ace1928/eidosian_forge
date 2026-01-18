from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml
from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader, parse_native_yaml
from torchgen.model import (
from torchgen.utils import NamespaceHelper
def parse_et_yaml_struct(es: object) -> ETKernelIndex:
    """Given a loaded yaml representing a list of operators, for each op extract the mapping
    of `kernel keys` to `BackendMetadata` (the latter representing the kernel instance
    that should be used by the kernel key).
    """
    indices: Dict[OperatorName, Dict[ETKernelKey, BackendMetadata]] = {}
    for ei in es:
        e = ei.copy()
        funcs = e.pop('func')
        assert isinstance(funcs, str), f'not a str: {funcs}'
        namespace_helper = NamespaceHelper.from_namespaced_entity(namespaced_entity=funcs, max_level=1)
        opname = FunctionSchema.parse(namespace_helper.entity_name).name
        assert opname not in indices, f'Duplicate func found in yaml: {opname} already'
        if len((index := parse_from_yaml(e))) != 0:
            indices[opname] = index
    return ETKernelIndex(indices)