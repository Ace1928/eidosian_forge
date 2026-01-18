import inspect
import logging
import numpy as np
import torch
import torch.utils._pytree as pytree
import pennylane as qml
def new_apply(*inp):
    out_struct_holder = []
    flat_out = orig_apply(out_struct_holder, *inp)
    return pytree.tree_unflatten(flat_out, out_struct_holder[0])