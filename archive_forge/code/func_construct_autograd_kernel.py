import torch
import torch.utils._pytree as pytree
from collections import namedtuple
import functools
def construct_autograd_kernel(schema, output_differentiability, custom_op, op_overload, save_for_backward_fn, backward_fn):

    def apply(*args):
        flat_args, spec = pytree.tree_flatten(args)
        out_spec = None

        def forward(ctx, *flat_args):
            ctx.set_materialize_grads(True)
            args = pytree.tree_unflatten(list(flat_args), spec)
            with torch._C._AutoDispatchBelowAutograd():
                output = op_overload(*args)
            args_info = namedtuple_args(schema, pytree.tree_map(type, args))
            save_for_backward_fn_inputs = namedtuple_args(schema, args)
            to_save = save_for_backward_fn(save_for_backward_fn_inputs, output)
            save_pytree_for_backward(ctx, (to_save, args_info))
            mark_non_differentiable(ctx, output, output_differentiability)
            nonlocal out_spec
            flat_output, out_spec = pytree.tree_flatten(output)
            return tuple(flat_output)

        def backward(ctx, *flat_grad_output):
            assert out_spec is not None
            grads = pytree.tree_unflatten(list(flat_grad_output), out_spec)
            saved, args_info = unpack_saved(ctx)
            inner_ctx = object()
            if not isinstance(grads, tuple):
                grads = (grads,)
            grad_inputs_dict = backward_fn(inner_ctx, saved, *grads)
            validate_grad_inputs_dict(grad_inputs_dict, custom_op, args_info)
            return grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info)
        generated_cls = gen_autograd_function(custom_op._opname + '_customop', forward, backward)
        flat_output = generated_cls.apply(*flat_args)
        assert out_spec is not None
        return pytree.tree_unflatten(list(flat_output), out_spec)
    return apply