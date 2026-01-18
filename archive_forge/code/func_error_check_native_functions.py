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
def error_check_native_functions(funcs: Sequence[NativeFunction]) -> None:
    func_map: Dict[OperatorName, NativeFunction] = {}
    base_func_map: Dict[BaseOperatorName, List[NativeFunction]] = defaultdict(list)
    for f in funcs:
        func_map[f.func.name] = f
        base_func_map[f.func.name.name].append(f)
    for f in funcs:
        if f.structured_delegate is not None:
            delegate_func = func_map[f.structured_delegate]
            assert delegate_func.structured, f"{f.func.name} is marked as a structured_delegate pointing to {f.structured_delegate}, but {f.structured_delegate} is not marked as structured. Consider adding 'structured=True' to the delegated operator"
        if 'inplace_view' in f.tags and str(f.func.name) != 'resize_' and (str(f.func.name) != 'resize_as_'):
            base_name = f.func.name.name
            overload_name = f.func.name.overload_name
            assert base_name.inplace, f"{f.func.name} is marked with tag: inplace_view, but it doesn't follow the naming convention for inplace ops - the codegen expects the base name to have a trailing underscore. "
            out_of_place_base_name = BaseOperatorName(base_name.base, False, base_name.dunder_method)
            assert len(base_func_map[out_of_place_base_name]) > 0, f"{f.func.name} is marked with tag: inplace_view. The codegen expects there to be a corresponding out-of-place view op with the name '{base_name}' and matching schema, but it didn't find one. "