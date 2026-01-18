import inspect
import logging
import numpy as np
import torch
import torch.utils._pytree as pytree
import pennylane as qml
def new_forward(ctx, out_struct_holder, *inp):
    out = orig_fw(ctx, *inp)
    flat_out, out_struct = pytree.tree_flatten(out)
    ctx._out_struct = out_struct
    out_struct_holder.append(out_struct)
    return tuple(flat_out)