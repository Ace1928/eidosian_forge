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
def static_dispatch_ops_header(f: NativeFunction, backend_index: List[BackendIndex]) -> Optional[str]:
    if backend_index is None or f.manual_kernel_registration:
        return None
    output = []
    for index in backend_index:
        dispatch_key = get_static_dispatch_backend(f, index)
        if dispatch_key is not None:
            output.append(f'#include <ATen/ops/{f.root_name}_{dispatch_key.lower()}_dispatch.h>')
    return '\n'.join(output)