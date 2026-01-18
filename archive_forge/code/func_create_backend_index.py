import argparse
import os
import pathlib
import re
from collections import Counter, defaultdict, namedtuple
from typing import Dict, List, Optional, Sequence, Set, Union
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.dest as dest
from torchgen.api.types import DispatcherSignature
from torchgen.code_template import CodeTemplate
from torchgen.context import native_function_manager
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, context, FileManager, NamespaceHelper, Target
from torchgen.yaml_utils import YamlLoader
def create_backend_index(backend_ops: List[str], symint_ops: Set[str], dispatch_key: DispatchKey, *, use_out_as_primary: bool, use_device_guard: bool) -> BackendIndex:
    metadata: Dict[OperatorName, BackendMetadata] = {}
    for op in backend_ops:
        op_name = OperatorName.parse(op)
        assert op_name in native_functions_map, f'Found an invalid operator name: {op_name}'
        kernel_name = dispatcher.name(native_functions_map[op_name].func)
        if op in symint_ops:
            kernel_name += '_symint'
        m = BackendMetadata(kernel=kernel_name, structured=False, cpp_namespace=cpp_namespace)
        metadata[op_name] = m
    return BackendIndex(dispatch_key=dispatch_key, use_out_as_primary=use_out_as_primary, external=True, device_guard=use_device_guard, index=metadata)