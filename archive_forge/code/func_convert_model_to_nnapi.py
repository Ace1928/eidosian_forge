from typing import List, Optional
import torch
from torch.backends._nnapi.serializer import _NnapiSerializer
def convert_model_to_nnapi(model, inputs, serializer=None, return_shapes=None, use_int16_for_qint16=False, compilation_preference=ANEURALNETWORKS_PREFER_SUSTAINED_SPEED, relax_f32_to_f16=False):
    shape_compute_module, ser_model_tensor, used_weights, inp_mem_fmts, out_mem_fmts, retval_count = process_for_nnapi(model, inputs, serializer, return_shapes, use_int16_for_qint16)
    nnapi_model = NnapiModule(shape_compute_module, ser_model_tensor, used_weights, inp_mem_fmts, out_mem_fmts, compilation_preference, relax_f32_to_f16)

    class NnapiInterfaceWrapper(torch.nn.Module):
        """NNAPI list-ifying and de-list-ifying wrapper.

        NNAPI always expects a list of inputs and provides a list of outputs.
        This module allows us to accept inputs as separate arguments.
        It returns results as either a single tensor or tuple,
        matching the original module.
        """

        def __init__(self, mod):
            super().__init__()
            self.mod = mod
    wrapper_model_py = NnapiInterfaceWrapper(nnapi_model)
    wrapper_model = torch.jit.script(wrapper_model_py)
    arg_list = ', '.join((f'arg_{idx}' for idx in range(len(inputs))))
    if retval_count < 0:
        ret_expr = 'retvals[0]'
    else:
        ret_expr = ''.join((f'retvals[{idx}], ' for idx in range(retval_count)))
    wrapper_model.define(f'def forward(self, {arg_list}):\n    retvals = self.mod([{arg_list}])\n    return {ret_expr}\n')
    return wrapper_model