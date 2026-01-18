import copy
import itertools
import warnings
import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
from torch.ao.nn.intrinsic import _FusedModule
from torch.ao.quantization.quantization_mappings import (
from .utils import get_qparam_dict, has_no_children_ignoring_parametrizations
from torch.ao.quantization.stubs import DeQuantStub, QuantWrapper
from torch.ao.quantization.qconfig import (
from torch.nn.utils.parametrize import type_before_parametrizations
from torch.ao.quantization.observer import _is_activation_post_process
from torch.ao.quantization.observer import (   # noqa: F401
def quantize_dynamic(model, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False):
    """Converts a float model to dynamic (i.e. weights-only) quantized model.

    Replaces specified modules with dynamic weight-only quantized versions and output the quantized model.

    For simplest usage provide `dtype` argument that can be float16 or qint8. Weight-only quantization
    by default is performed for layers with large weights size - i.e. Linear and RNN variants.

    Fine grained control is possible with `qconfig` and `mapping` that act similarly to `quantize()`.
    If `qconfig` is provided, the `dtype` argument is ignored.

    Args:
        model: input model
        qconfig_spec: Either:

            - A dictionary that maps from name or type of submodule to quantization
              configuration, qconfig applies to all submodules of a given
              module unless qconfig for the submodules are specified (when the
              submodule already has qconfig attribute). Entries in the dictionary
              need to be QConfig instances.

            - A set of types and/or submodule names to apply dynamic quantization to,
              in which case the `dtype` argument is used to specify the bit-width

        inplace: carry out model transformations in-place, the original module is mutated
        mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced

    """
    torch._C._log_api_usage_once('quantization_api.quantize.quantize_dynamic')
    if qconfig_spec is None:
        if dtype == torch.qint8:
            qconfig_spec = {nn.Linear: default_dynamic_qconfig, nn.LSTM: default_dynamic_qconfig, nn.GRU: default_dynamic_qconfig, nn.LSTMCell: default_dynamic_qconfig, nn.RNNCell: default_dynamic_qconfig, nn.GRUCell: default_dynamic_qconfig}
        elif dtype == torch.float16:
            qconfig_spec = {nn.Linear: float16_dynamic_qconfig, nn.LSTM: float16_dynamic_qconfig, nn.GRU: float16_dynamic_qconfig, nn.LSTMCell: float16_dynamic_qconfig, nn.RNNCell: float16_dynamic_qconfig, nn.GRUCell: float16_dynamic_qconfig}
        elif dtype == torch.quint8:
            qconfig_spec = {nn.EmbeddingBag: float_qparams_weight_only_qconfig, nn.Embedding: float_qparams_weight_only_qconfig}
        elif dtype == torch.quint4x2:
            qconfig_spec = {nn.EmbeddingBag: float_qparams_weight_only_qconfig_4bit}
        else:
            raise ValueError(f"Don't know how to quantize with default settings for {dtype}. Provide full qconfig please")
    elif isinstance(qconfig_spec, set):
        if dtype is torch.qint8:
            default_qconfig = default_dynamic_qconfig
        elif dtype is torch.float16:
            default_qconfig = float16_dynamic_qconfig
        elif dtype is torch.quint8:
            default_qconfig = float_qparams_weight_only_qconfig
        elif dtype is torch.quint4x2:
            default_qconfig = float_qparams_weight_only_qconfig_4bit
        else:
            raise RuntimeError('Unknown dtype specified for quantize_dynamic: ', str(dtype))
        qconfig_spec = dict(zip(qconfig_spec, itertools.repeat(default_qconfig)))
    if mapping is None:
        mapping = get_default_dynamic_quant_module_mappings()
    if not inplace:
        model = copy.deepcopy(model)
    model.eval()
    propagate_qconfig_(model, qconfig_spec)
    convert(model, mapping, inplace=True)
    return model