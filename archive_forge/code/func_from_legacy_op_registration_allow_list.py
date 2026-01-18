from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import yaml
from torchgen.model import NativeFunction
from torchgen.selective_build.operator import (
@staticmethod
def from_legacy_op_registration_allow_list(allow_list: Set[str], is_root_operator: bool, is_used_for_training: bool) -> 'SelectiveBuilder':
    operators = {}
    for op in allow_list:
        operators[op] = {'name': op, 'is_root_operator': is_root_operator, 'is_used_for_training': is_used_for_training, 'include_all_overloads': True}
    return SelectiveBuilder.from_yaml_dict({'operators': operators, 'include_all_non_op_selectives': True})