from typing import List, Optional
import torch
from torch.backends._nnapi.serializer import _NnapiSerializer
def process_for_nnapi(model, inputs, serializer=None, return_shapes=None, use_int16_for_qint16=False):
    model = torch.jit.freeze(model)
    if isinstance(inputs, torch.Tensor):
        inputs = [inputs]
    serializer = serializer or _NnapiSerializer(config=None, use_int16_for_qint16=use_int16_for_qint16)
    ser_model, used_weights, inp_mem_fmts, out_mem_fmts, shape_compute_lines, retval_count = serializer.serialize_model(model, inputs, return_shapes)
    ser_model_tensor = torch.tensor(ser_model, dtype=torch.int32)

    class ShapeComputeModule(torch.nn.Module):
        """Code-gen-ed module for tensor shape computation.

        module.prepare will mutate ser_model according to the computed operand
        shapes, based on the shapes of args.  Returns a list of output templates.
        """
        pass
    shape_compute_module = torch.jit.script(ShapeComputeModule())
    real_shape_compute_lines = ['def prepare(self, ser_model: torch.Tensor, args: List[torch.Tensor]) -> List[torch.Tensor]:\n'] + [f'    {line}\n' for line in shape_compute_lines]
    shape_compute_module.define(''.join(real_shape_compute_lines))
    return (shape_compute_module, ser_model_tensor, used_weights, inp_mem_fmts, out_mem_fmts, retval_count)