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
def compute_argument_yaml(a: Argument, *, schema_order: bool, kwarg_only_set: Set[str], out_arg_set: Set[str], name_to_field_name: Dict[str, str]) -> object:
    arg: Dict[str, object] = {'annotation': str(a.annotation) if a.annotation else None, 'dynamic_type': dynamic_type(a.type), 'is_nullable': a.type.is_nullable(), 'name': a.name, 'type': cpp.argument_type(a, binds='__placeholder__', symint=False).cpp_type()}
    if a.default is not None:
        arg['default'] = pythonify_default(cpp.default_expr(a.default, a.type, symint=False))
    if a.name in kwarg_only_set:
        arg['kwarg_only'] = True
    if a.name in out_arg_set:
        arg['output'] = True
        arg['allocate'] = True
        if a.name in name_to_field_name:
            arg['field_name'] = name_to_field_name[a.name]
    l = a.type.is_list_like()
    if l is not None and l.size is not None and (str(l.elem) != 'bool'):
        arg['size'] = l.size
    return arg