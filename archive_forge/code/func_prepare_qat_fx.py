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
def prepare_qat_fx(model: torch.nn.Module, qconfig_mapping: Union[QConfigMapping, Dict[str, Any]], example_inputs: Tuple[Any, ...], prepare_custom_config: Union[PrepareCustomConfig, Dict[str, Any], None]=None, backend_config: Union[BackendConfig, Dict[str, Any], None]=None) -> GraphModule:
    """ Prepare a model for quantization aware training

    Args:
      * `model` (torch.nn.Module): torch.nn.Module model
      * `qconfig_mapping` (QConfigMapping): see :func:`~torch.ao.quantization.prepare_fx`
      * `example_inputs` (Tuple[Any, ...]): see :func:`~torch.ao.quantization.prepare_fx`
      * `prepare_custom_config` (PrepareCustomConfig): see :func:`~torch.ao.quantization.prepare_fx`
      * `backend_config` (BackendConfig): see :func:`~torch.ao.quantization.prepare_fx`

    Return:
      A GraphModule with fake quant modules (configured by qconfig_mapping and backend_config), ready for
      quantization aware training

    Example::

        import torch
        from torch.ao.quantization import get_default_qat_qconfig_mapping
        from torch.ao.quantization.quantize_fx import prepare_qat_fx

        class Submodule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
            def forward(self, x):
                x = self.linear(x)
                return x

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)
                self.sub = Submodule()

            def forward(self, x):
                x = self.linear(x)
                x = self.sub(x) + x
                return x

        # initialize a floating point model
        float_model = M().train()
        # (optional, but preferred) load the weights from pretrained model
        # float_model.load_weights(...)

        # define the training loop for quantization aware training
        def train_loop(model, train_data):
            model.train()
            for image, target in data_loader:
                ...

        # qconfig is the configuration for how we insert observers for a particular
        # operator
        # qconfig = get_default_qconfig("fbgemm")
        # Example of customizing qconfig:
        # qconfig = torch.ao.quantization.QConfig(
        #    activation=FakeQuantize.with_args(observer=MinMaxObserver.with_args(dtype=torch.qint8)),
        #    weight=FakeQuantize.with_args(observer=MinMaxObserver.with_args(dtype=torch.qint8)))
        # `activation` and `weight` are constructors of observer module

        # qconfig_mapping is a collection of quantization configurations, user can
        # set the qconfig for each operator (torch op calls, functional calls, module calls)
        # in the model through qconfig_mapping
        # the following call will get the qconfig_mapping that works best for models
        # that target "fbgemm" backend
        qconfig_mapping = get_default_qat_qconfig("fbgemm")

        # We can customize qconfig_mapping in different ways, please take a look at
        # the docstring for :func:`~torch.ao.quantization.prepare_fx` for different ways
        # to configure this

        # example_inputs is a tuple of inputs, that is used to infer the type of the
        # outputs in the model
        # currently it's not used, but please make sure model(*example_inputs) runs
        example_inputs = (torch.randn(1, 3, 224, 224),)

        # TODO: add backend_config after we split the backend_config for fbgemm and qnnpack
        # e.g. backend_config = get_default_backend_config("fbgemm")
        # `prepare_qat_fx` inserts observers in the model based on qconfig_mapping and
        # backend_config, if the configuration for an operator in qconfig_mapping
        # is supported in the backend_config (meaning it's supported by the target
        # hardware), we'll insert fake_quantize modules according to the qconfig_mapping
        # otherwise the configuration in qconfig_mapping will be ignored
        # see :func:`~torch.ao.quantization.prepare_fx` for a detailed explanation of
        # how qconfig_mapping interacts with backend_config
        prepared_model = prepare_qat_fx(float_model, qconfig_mapping, example_inputs)
        # Run training
        train_loop(prepared_model, train_loop)

    """
    torch._C._log_api_usage_once('quantization_api.quantize_fx.prepare_qat_fx')
    return _prepare_fx(model, qconfig_mapping, True, example_inputs, prepare_custom_config, backend_config=backend_config)