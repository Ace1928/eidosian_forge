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
def dict_representer(dumper: Any, data: Any) -> Any:
    return dumper.represent_dict(data.items())