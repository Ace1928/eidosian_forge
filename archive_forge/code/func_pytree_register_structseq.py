import torch
import inspect
def pytree_register_structseq(cls):

    def structseq_flatten(structseq):
        return (list(structseq), None)

    def structseq_unflatten(values, context):
        return cls(values)
    torch.utils._pytree.register_pytree_node(cls, structseq_flatten, structseq_unflatten)