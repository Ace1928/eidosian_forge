from collections import defaultdict, namedtuple
from typing import Any, Dict, List, Optional, Set, Tuple
import yaml
from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader, parse_native_yaml
from torchgen.model import (
from torchgen.utils import NamespaceHelper
def parse_from_yaml(ei: Dict[str, object]) -> Dict[ETKernelKey, BackendMetadata]:
    """Given a loaded yaml representing kernel assignment information, extract the
    mapping from `kernel keys` to `BackendMetadata` (the latter representing the kernel instance)

    Args:
        ei: Dict keys {kernels, type_alias, dim_order_alias}
            See ETKernelKey for description of arguments
    """
    e = ei.copy()
    if (kernels := e.pop('kernels', None)) is None:
        return {}
    type_alias: Dict[str, List[str]] = e.pop('type_alias', {})
    dim_order_alias: Dict[str, List[str]] = e.pop('dim_order_alias', {})
    dim_order_alias.pop('__line__', None)
    kernel_mapping: Dict[ETKernelKey, BackendMetadata] = {}
    for entry in kernels:
        arg_meta = entry.get('arg_meta')
        if arg_meta is not None:
            arg_meta.pop('__line__')
        kernel_name = entry.get('kernel_name')
        namespace_helper = NamespaceHelper.from_namespaced_entity(kernel_name, max_level=3)
        kernel_namespace = namespace_helper.get_cpp_namespace(default='at')
        backend_metadata = BackendMetadata(kernel=namespace_helper.entity_name, structured=False, cpp_namespace=kernel_namespace + '::native')
        kernel_keys = [ETKernelKey((), default=True)] if arg_meta is None else ETKernelKey.gen_from_yaml(arg_meta, type_alias, dim_order_alias)
        for kernel_key in kernel_keys:
            assert kernel_key not in kernel_mapping, 'Duplicate kernel key: ' + str(kernel_key) + ' ' + str(e)
            kernel_mapping[kernel_key] = backend_metadata
    return kernel_mapping