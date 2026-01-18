import inspect
from typing import Any, Dict, List, Optional, Union
import torch.nn
from . import utils, variables
from .bytecode_transformation import (
from .codegen import PyCodegen
from .exc import unimplemented
from .source import LocalSource, Source
from .utils import nn_module_new, object_new
from .variables.base import (
def store_attr(self, item: VariableTracker, name: str, value: VariableTracker):
    assert self.is_attribute_mutation(item)
    self.check_allowed_side_effect(item)
    if item.mutable_local not in self.store_attr_mutations:
        self.store_attr_mutations[item.mutable_local] = {}
    self.store_attr_mutations[item.mutable_local][name] = value