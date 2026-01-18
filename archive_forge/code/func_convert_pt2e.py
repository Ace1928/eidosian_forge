import torch
from torch.fx import GraphModule
from torch.fx import Node
from .pt2e.prepare import prepare
from .pt2e.qat_utils import (
from .pt2e.utils import (
from .pt2e.representation import reference_representation_rewrite
from .quantize_fx import _convert_to_reference_decomposed_fx
from torch.ao.quantization.quantizer import (  # noqa: F401
from torch.fx.passes.infra.pass_manager import PassManager
from torch.ao.quantization.pt2e.duplicate_dq_pass import DuplicateDQPass
from torch.ao.quantization.pt2e.port_metadata_pass import PortNodeMetaForQDQ
from torch._inductor.constant_folding import constant_fold
def convert_pt2e(model: GraphModule, use_reference_representation: bool=False, fold_quantize: bool=False) -> GraphModule:
    """Convert a calibrated/trained model to a quantized model

    Args:
      * `model` (torch.fx.GraphModule): calibrated/trained model
      * `use_reference_representation` (bool): boolean flag to indicate whether to produce referece representation or not
      * `fold_quantize` (bool): boolean flag to indicate whether fold the quantize op or not

    Note: please set `fold_quantize` to True whenever you can, we'll deprecate this flag and
    make True the default option in the future, to make sure the change doesn't break BC for you, it's
    better to set the flag to True now.

    Returns:
        quantized model, either in q/dq representation or reference representation

    Example::

        # prepared_model: the model produced by `prepare_pt2e`/`prepare_qat_pt2e` and calibration/training
        # `convert_pt2e` produces a quantized model that represents quantized computation with
        # quantize dequantize ops and fp32 ops by default.
        # Please refer to
        # https://pytorch.org/tutorials/prototype/pt2e_quant_ptq_static.html#convert-the-calibrated-model-to-a-quantized-model
        # for detailed explanation of output quantized model
        quantized_model = convert_pt2e(prepared_model)

    """
    torch._C._log_api_usage_once('quantization_api.quantize_pt2e.convert_pt2e')
    original_graph_meta = model.meta
    model = _convert_to_reference_decomposed_fx(model)
    model = _fold_conv_bn_qat(model)
    pm = PassManager([DuplicateDQPass()])
    model = pm(model).graph_module
    pm = PassManager([PortNodeMetaForQDQ()])
    model = pm(model).graph_module
    if fold_quantize:
        constant_fold(model, _quant_node_constraint)
    if use_reference_representation:
        model = reference_representation_rewrite(model)
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model