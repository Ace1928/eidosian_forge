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
def prepare_qat_pt2e(model: GraphModule, quantizer: Quantizer) -> GraphModule:
    """Prepare a model for quantization aware training

    Args:
      * `model` (torch.fx.GraphModule): see :func:`~torch.ao.quantization.quantize_pt2e.prepare_pt2e`
      * `quantizer`: see :func:`~torch.ao.quantization.quantize_pt2e.prepare_pt2e`

    Return:
      A GraphModule with fake quant modules (based on quantizer annotation), ready for
      quantization aware training

    Example::
        import torch
        from torch.ao.quantization.quantize_pt2e import prepare_qat_pt2e
        from torch._export import capture_pre_autograd_graph
        from torch.ao.quantization.quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 10)

           def forward(self, x):
               return self.linear(x)

        # initialize a floating point model
        float_model = M().eval()

        # define the training loop for quantization aware training
        def train_loop(model, train_data):
            model.train()
            for image, target in data_loader:
                ...

        # Step 1. program capture
        # NOTE: this API will be updated to torch.export API in the future, but the captured
        # result shoud mostly stay the same
        m = capture_pre_autograd_graph(m, *example_inputs)
        # we get a model with aten ops

        # Step 2. quantization
        # backend developer will write their own Quantizer and expose methods to allow
        # users to express how they
        # want the model to be quantized
        quantizer = XNNPACKQuantizer().set_global(get_symmetric_quantization_config())
        m = prepare_qat_pt2e(m, quantizer)

        # run quantization aware training
        train_loop(prepared_model, train_loop)

    """
    torch._C._log_api_usage_once('quantization_api.quantize_pt2e.prepare_qat_pt2e')
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    quantizer.transform_for_annotation(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    _fuse_conv_bn_qat(model)
    model = prepare(model, node_name_to_scope, is_qat=True)
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model