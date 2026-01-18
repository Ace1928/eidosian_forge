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
def prepare_pt2e(model: GraphModule, quantizer: Quantizer) -> GraphModule:
    """Prepare a model for post training quantization

    Args:
      * `model` (torch.fx.GraphModule): a model captured by `torch.export` API
        in the short term we are using `torch._export.capture_pre_autograd_graph`,
        in the long term we'll migrate to some `torch.export` API
      * `quantizer`: A backend specific quantizer that conveys how user want the
        model to be quantized. Tutorial for how to write a quantizer can be found here:
        https://pytorch.org/tutorials/prototype/pt2e_quantizer.html

    Return:
      A GraphModule with observer (based on quantizer annotation), ready for calibration

    Example::

        import torch
        from torch.ao.quantization.quantize_pt2e import prepare_pt2e
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

        # define calibration function
        def calibrate(model, data_loader):
            model.eval()
            with torch.no_grad():
                for image, target in data_loader:
                    model(image)

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
        m = prepare_pt2e(m, quantizer)

        # run calibration
        # calibrate(m, sample_inference_data)
    """
    torch._C._log_api_usage_once('quantization_api.quantize_pt2e.prepare_pt2e')
    original_graph_meta = model.meta
    node_name_to_scope = _get_node_name_to_scope(model)
    _fuse_conv_bn_(model)
    quantizer.transform_for_annotation(model)
    quantizer.annotate(model)
    quantizer.validate(model)
    model = prepare(model, node_name_to_scope, is_qat=False)
    model.meta.update(original_graph_meta)
    model = _disallow_eval_train(model)
    return model