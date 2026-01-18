import argparse
import os
import pathlib
import re
from collections import Counter, namedtuple
from typing import (
import yaml
import torchgen.dest as dest
from torchgen.api.lazy import setValueT
from torchgen.api.types import BaseCppType
from torchgen.dest.lazy_ir import GenLazyIR, GenLazyNativeFuncDefinition, GenTSLazyIR
from torchgen.gen import get_grouped_native_functions, parse_native_yaml
from torchgen.model import NativeFunction, NativeFunctionsGroup, OperatorName
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import concatMap, FileManager, NamespaceHelper
from torchgen.yaml_utils import YamlLoader
from .gen_backend_stubs import (
def parse_native_functions_keys(backend_yaml_path: str, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]]) -> Tuple[List[OperatorName], List[Any], List[OperatorName]]:
    native_functions_map: Dict[OperatorName, NativeFunction] = {f.func.name: f for f in concatMap(lambda f: [f] if isinstance(f, NativeFunction) else list(f.functions()), grouped_native_functions)}
    with open(backend_yaml_path) as f:
        yaml_values = yaml.load(f, Loader=YamlLoader)
    assert isinstance(yaml_values, dict)
    full_codegen = yaml_values.pop('full_codegen', [])
    non_native = yaml_values.pop('non_native', [])
    ir_gen = yaml_values.pop('ir_gen', [])
    assert isinstance(full_codegen, list)
    assert isinstance(non_native, list)
    assert isinstance(ir_gen, list)
    full_codegen_opnames = [OperatorName.parse(name) for name in full_codegen]
    ir_gen_opnames = [OperatorName.parse(name) for name in ir_gen]
    return (full_codegen_opnames, non_native, ir_gen_opnames)