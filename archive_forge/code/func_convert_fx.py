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
def convert_fx(graph_module: GraphModule, convert_custom_config: Union[ConvertCustomConfig, Dict[str, Any], None]=None, _remove_qconfig: bool=True, qconfig_mapping: Union[QConfigMapping, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    """ Convert a calibrated or trained model to a quantized model

    Args:
        * `graph_module` (torch.fx.GraphModule): A prepared and calibrated/trained model (GraphModule)

        * `convert_custom_config` (ConvertCustomConfig): custom configurations for convert function.
            See :class:`~torch.ao.quantization.fx.custom_config.ConvertCustomConfig` for more details

        * `_remove_qconfig` (bool): Option to remove the qconfig attributes in the model after convert.

        * `qconfig_mapping` (QConfigMapping): config for specifying how to convert a model for quantization.

           The keys must include the ones in the qconfig_mapping passed to `prepare_fx` or `prepare_qat_fx`,
           with the same values or `None`. Additional keys can be specified with values set to `None`.

          For each entry whose value is set to None, we skip quantizing that entry in the model::

            qconfig_mapping = QConfigMapping
                .set_global(qconfig_from_prepare)
                .set_object_type(torch.nn.functional.add, None)  # skip quantizing torch.nn.functional.add
                .set_object_type(torch.nn.functional.linear, qconfig_from_prepare)
                .set_module_name("foo.bar", None)  # skip quantizing module "foo.bar"

         * `backend_config` (BackendConfig): A configuration for the backend which describes how
            operators should be quantized in the backend, this includes quantization
            mode support (static/dynamic/weight_only), dtype support (quint8/qint8 etc.),
            observer placement for each operators and fused operators.
            See :class:`~torch.ao.quantization.backend_config.BackendConfig` for more details

    Return:
        A quantized model (torch.nn.Module)

    Example::

        # prepared_model: the model after prepare_fx/prepare_qat_fx and calibration/training
        # convert_fx converts a calibrated/trained model to a quantized model for the
        # target hardware, this includes converting the model first to a reference
        # quantized model, and then lower the reference quantized model to a backend
        # Currently, the supported backends are fbgemm (onednn), qnnpack (xnnpack) and
        # they share the same set of quantized operators, so we are using the same
        # lowering procedure
        #
        # backend_config defines the corresponding reference quantized module for
        # the weighted modules in the model, e.g. nn.Linear
        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        quantized_model = convert_fx(prepared_model)

    """
    torch._C._log_api_usage_once('quantization_api.quantize_fx.convert_fx')
    return _convert_fx(graph_module, is_reference=False, convert_custom_config=convert_custom_config, _remove_qconfig=_remove_qconfig, qconfig_mapping=qconfig_mapping, backend_config=backend_config)