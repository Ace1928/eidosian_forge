from __future__ import annotations
import dataclasses
import enum
import logging
from typing import FrozenSet, List, Mapping, Optional, Sequence, Tuple
from torch.onnx._internal.diagnostics.infra import formatter, sarif
@classmethod
def custom_collection_from_list(cls, new_collection_class_name: str, rules: Sequence[Rule]) -> RuleCollection:
    """Creates a custom class inherited from RuleCollection with the list of rules."""
    return dataclasses.make_dataclass(new_collection_class_name, [(formatter.kebab_case_to_snake_case(rule.name), type(rule), dataclasses.field(default=rule)) for rule in rules], bases=(cls,))()