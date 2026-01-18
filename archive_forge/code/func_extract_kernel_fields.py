from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml
from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader, parse_native_yaml
from torchgen.model import (
from torchgen.utils import NamespaceHelper
def extract_kernel_fields(es: object) -> Dict[OperatorName, Dict[str, Any]]:
    """Given a loaded yaml representing a list of operators, extract the
    kernel key related fields indexed by the operator name.
    """
    fields: Dict[OperatorName, Dict[str, Any]] = defaultdict(dict)
    for ei in es:
        funcs = ei.get('func')
        assert isinstance(funcs, str), f'not a str: {funcs}'
        namespace_helper = NamespaceHelper.from_namespaced_entity(namespaced_entity=funcs, max_level=1)
        opname = FunctionSchema.parse(namespace_helper.entity_name).name
        for field in ET_FIELDS:
            if (value := ei.get(field)) is not None:
                fields[opname][field] = value
    return fields