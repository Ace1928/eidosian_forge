import argparse
import functools
import json
import os
import pathlib
from collections import defaultdict, namedtuple, OrderedDict
from dataclasses import dataclass, field
from typing import (
import yaml
import torchgen.api.dispatcher as dispatcher
import torchgen.api.meta as meta
import torchgen.api.native as native
import torchgen.api.structured as structured
import torchgen.dest as dest
from torchgen.api import cpp
from torchgen.api.translate import translate
from torchgen.api.types import (
from torchgen.context import (
from torchgen.gen_functionalization_type import (
from torchgen.gen_vmap_plumbing import gen_all_vmap_plumbing
from torchgen.model import (
from torchgen.native_function_generation import (
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
from torchgen.yaml_utils import YamlDumper, YamlLoader
def generate_static_dispatch_backend_call(sig: Union[CppSignature, DispatcherSignature], f: NativeFunction, backend_index: BackendIndex) -> str:
    cpp_sigs = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    if sig.symint and f.func.has_symint():
        cpp_sig = cpp_sigs.symint_signature
    else:
        cpp_sig = cpp_sigs.signature
    assert cpp_sig is not None
    name = cpp_sig.name()
    exprs = translate_args(sig, cpp_sig)
    backend_metadata = backend_index.get_kernel(f)
    kernel_ns = backend_metadata.cpp_namespace if backend_metadata and backend_metadata.cpp_namespace else DEFAULT_KERNEL_NAMESPACE
    ns = kernel_ns.replace('::native', '')
    return f'return {ns}::{backend_index.dispatch_key.lower()}::{name}({exprs});'