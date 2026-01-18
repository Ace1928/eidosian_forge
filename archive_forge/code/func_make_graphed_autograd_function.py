import gc
import torch
from torch.utils import _pytree
from ._utils import _dummy_type
from torch._C import (  # noqa: F401
def make_graphed_autograd_function(fwd_graph, bwd_graph, module_params, len_user_args, output_unflatten_spec, static_input_surface, static_outputs, static_grad_outputs, static_grad_inputs):

    class Graphed(torch.autograd.Function):

        @staticmethod
        def forward(ctx, *inputs):
            for i in range(len_user_args):
                if static_input_surface[i].data_ptr() != inputs[i].data_ptr():
                    static_input_surface[i].copy_(inputs[i])
            fwd_graph.replay()
            assert isinstance(static_outputs, tuple)
            return tuple((o.detach() for o in static_outputs))

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx, *grads):
            assert len(grads) == len(static_grad_outputs)
            for g, grad in zip(static_grad_outputs, grads):
                if g is not None:
                    if g.data_ptr() != grad.data_ptr():
                        g.copy_(grad)
            bwd_graph.replay()
            assert isinstance(static_grad_inputs, tuple)
            return tuple((b.detach() if b is not None else b for b in static_grad_inputs))

    def functionalized(*user_args):
        flatten_user_args = _pytree.arg_tree_leaves(*user_args)
        out = Graphed.apply(*tuple(flatten_user_args) + module_params)
        return _pytree.tree_unflatten(out, output_unflatten_spec)
    return functionalized