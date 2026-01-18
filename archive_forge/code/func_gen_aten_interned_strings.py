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
def gen_aten_interned_strings() -> Dict[str, str]:
    attrs = set()
    names = set()
    for func in native_functions:
        names.add(str(func.func.name.name))
        names.add(func.func.name.name.base)
        for arg in func.func.schema_order_arguments():
            attrs.add(arg.name)
    names -= {'and', 'and_eq', 'bitand', 'bitor', 'compl', 'not', 'not_eq', 'or', 'or_eq', 'xor', 'xor_eq'}
    return {'aten_symbols': ' \\\n'.join([f'_(aten, {name})' for name in sorted(names)]), 'attr_symbols': ' \\\n'.join([f'_(attr, {name})' for name in sorted(attrs)])}