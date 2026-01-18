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
def get_native_function_definitions(*, fm: FileManager, grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]], dispatch_key: DispatchKey, backend_idx: BackendIndex, selector: SelectiveBuilder, rocm: bool, symint: bool, skip_dispatcher_op_registration: bool, gen_dispatch_helpers: bool) -> List[str]:
    definitions: List[str] = []
    ns_definitions: Dict[str, List[str]] = defaultdict(list)
    anonymous_definitions: Dict[str, List[str]] = defaultdict(list)
    registrations: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    newline = '\n'
    ns_gen = dest.RegisterDispatchKey(backend_idx, Target.NAMESPACED_DEFINITION, selector, rocm=rocm, symint=symint, class_method_name=None, skip_dispatcher_op_registration=skip_dispatcher_op_registration)
    anonymous_gen = dest.RegisterDispatchKey(backend_idx, Target.ANONYMOUS_DEFINITION, selector, rocm=rocm, symint=symint, class_method_name=None, skip_dispatcher_op_registration=skip_dispatcher_op_registration)
    reg_gen = dest.RegisterDispatchKey(backend_idx, Target.REGISTRATION, selector, rocm=rocm, symint=symint, class_method_name=None, skip_dispatcher_op_registration=skip_dispatcher_op_registration)
    for f in grouped_native_functions:
        kernel_namespace = get_kernel_namespace(f=f, backend_idx=backend_idx).replace('::native', '')
        ns_definitions[kernel_namespace].extend(ns_gen(f))
        anonymous_definitions[kernel_namespace].extend(anonymous_gen(f))
        namespace = f.namespace if isinstance(f, NativeFunction) else f.functional.namespace
        if namespace not in registrations[kernel_namespace]:
            registrations[kernel_namespace] = defaultdict(list)
        registrations[kernel_namespace][namespace].extend(reg_gen(f))
    for kernel_namespace in ns_definitions:
        if len(ns_definitions[kernel_namespace]) == 0:
            continue
        ns_helper = NamespaceHelper(namespace_str=kernel_namespace)
        registration_body = ''
        for namespace in registrations[kernel_namespace]:
            if not registrations[kernel_namespace][namespace]:
                continue
            registration_body += f'\nTORCH_LIBRARY_IMPL({namespace}, {dispatch_key}, m) {{\n    {newline.join(registrations[kernel_namespace][namespace])}\n}};'
        definitions.extend(fm.substitute_with_template('RegisterDispatchDefinitions.ini', lambda: {'ns_prologue': ns_helper.prologue, 'ns_epilogue': ns_helper.epilogue, 'dispatch_helpers': dest.gen_registration_helpers(backend_idx) if gen_dispatch_helpers else [], 'dispatch_anonymous_definitions': anonymous_definitions[kernel_namespace], 'static_init_dispatch_registrations': '' if skip_dispatcher_op_registration else registration_body, 'deferred_dispatch_registrations': '', 'dispatch_namespace': dispatch_key.lower(), 'dispatch_namespaced_definitions': ns_definitions[kernel_namespace]}).split(newline))
    return definitions