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
@with_native_function
def compute_declaration_yaml(f: NativeFunction) -> object:
    returns, name_to_field_name = compute_returns_yaml(f)
    kwarg_only_set = {a.name for a in f.func.arguments.flat_kwarg_only}
    out_arg_set = {a.name for a in f.func.arguments.out}
    sig_group = CppSignatureGroup.from_native_function(f, method=False, fallback_binding=False)
    cpp_args = sig_group.signature.arguments()
    arguments = [compute_cpp_argument_yaml(cpp_a, schema_order=False, kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name) for cpp_a in cpp_args]
    schema_order_jit_arguments = list(f.func.schema_order_arguments())
    schema_order_arguments = [compute_argument_yaml(a, schema_order=True, kwarg_only_set=kwarg_only_set, out_arg_set=out_arg_set, name_to_field_name=name_to_field_name) for a in schema_order_jit_arguments]
    cpp_schema_order_types = [r.type for a in schema_order_jit_arguments for r in cpp.argument(a, method=False, cpp_no_default_args=set(), faithful=False, symint=False, has_tensor_options=False)]
    cpp_returns = cpp.returns_type(f.func.returns, symint=False).cpp_type()
    schema_order_cpp_signature = f'{cpp_returns} ({', '.join(cpp_schema_order_types)})'
    is_factory_method = any((isinstance(a.argument, TensorOptionsArguments) for a in cpp_args)) and Variant.method not in f.variants
    return OrderedDict([('name', cpp.name(f.func)), ('operator_name', str(f.func.name.name)), ('overload_name', str(f.func.name.overload_name)), ('manual_kernel_registration', f.manual_kernel_registration), ('category_override', f.category_override if f.category_override is not None else ''), ('schema_string', f'aten::{f.func}'), ('arguments', arguments), ('schema_order_cpp_signature', schema_order_cpp_signature), ('schema_order_arguments', schema_order_arguments), ('method_of', compute_method_of_yaml(f.variants)), ('mode', 'native'), ('python_module', '' if f.python_module is None else f.python_module), ('returns', returns), ('inplace', f.func.name.name.inplace), ('is_factory_method', is_factory_method), ('abstract', f.is_abstract), ('device_guard', f.device_guard), ('with_gil', False), ('deprecated', False), ('has_math_kernel', f.has_composite_implicit_autograd_kernel)])