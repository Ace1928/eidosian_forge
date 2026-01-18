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
def fuse_fx(model: torch.nn.Module, fuse_custom_config: Union[FuseCustomConfig, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    """ Fuse modules like conv+bn, conv+bn+relu etc, model must be in eval mode.
    Fusion rules are defined in torch.ao.quantization.fx.fusion_pattern.py

    Args:

        * `model` (torch.nn.Module): a torch.nn.Module model
        * `fuse_custom_config` (FuseCustomConfig): custom configurations for fuse_fx.
            See :class:`~torch.ao.quantization.fx.custom_config.FuseCustomConfig` for more details
    Example::

        from torch.ao.quantization import fuse_fx
        m = Model().eval()
        m = fuse_fx(m)

    """
    if fuse_custom_config is None:
        fuse_custom_config = FuseCustomConfig()
    if isinstance(fuse_custom_config, Dict):
        warnings.warn('Passing a fuse_custom_config_dict to fuse is deprecated and will not be supported in a future version. Please pass in a FuseCustomConfig instead.')
        fuse_custom_config = FuseCustomConfig.from_dict(fuse_custom_config)
    torch._C._log_api_usage_once('quantization_api.quantize_fx.fuse_fx')
    preserved_attr_names = fuse_custom_config.preserved_attributes
    preserved_attrs = {attr: getattr(model, attr) for attr in preserved_attr_names if hasattr(model, attr)}
    graph_module = torch.fx.symbolic_trace(model)
    _attach_meta_to_node_if_not_exist(graph_module)
    graph_module = _fuse_fx(graph_module, False, fuse_custom_config, backend_config)
    attach_preserved_attrs_to_model(graph_module, preserved_attrs)
    return graph_module