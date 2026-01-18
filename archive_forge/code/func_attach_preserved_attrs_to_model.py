from typing import Any, Dict, Optional, Tuple, Union
import warnings
import torch
import copy
from torch.fx import GraphModule
from torch.fx.graph_module import _USER_PRESERVED_ATTRIBUTES_KEY
from .fx.tracer import QuantizationTracer
from .fx.tracer import (  # noqa: F401
from .fx.fuse import fuse  # noqa: F401
from .fx.prepare import prepare  # noqa: F401
from .fx.convert import convert
from .backend_config import (  # noqa: F401
from .fx.graph_module import ObservedGraphModule  # noqa: F401
from .fx.custom_config import (
from .fx.utils import get_custom_module_class_keys  # noqa: F401
from .fx.utils import get_skipped_module_name_and_classes
from .qconfig_mapping import QConfigMapping
def attach_preserved_attrs_to_model(model: Union[GraphModule, torch.nn.Module], preserved_attrs: Dict[str, Any]) -> None:
    """ Store preserved attributes to the model.meta so that it can be preserved during deepcopy
    """
    model.meta[_USER_PRESERVED_ATTRIBUTES_KEY] = copy.copy(preserved_attrs)
    for attr_name, attr in model.meta[_USER_PRESERVED_ATTRIBUTES_KEY].items():
        setattr(model, attr_name, attr)