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
def parse_tags_yaml_struct(es: object, path: str='<stdin>') -> Set[str]:
    assert isinstance(es, list)
    rs: Set[str] = set()
    for e in es:
        assert isinstance(e.get('__line__'), int), e
        loc = Location(path, e['__line__'])
        tags = e.get('tag')
        with context(lambda: f'in {loc}:\n  {tags}'):
            e_i = e.copy()
            name = e_i.pop('tag')
            desc = e_i.pop('desc', '')
            assert desc != ''
            rs.add(name)
    return rs