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
@staticmethod
def cls_supports_mutation_side_effects(cls):
    return inspect.getattr_static(cls, '__setattr__', None) in (object.__setattr__, torch.nn.Module.__setattr__)