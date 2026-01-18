import torch
from torch._export.db.case import export_case
from torch.export import Dim
@export_case(example_inputs=(x,), tags={'torch.dynamic-shape'}, dynamic_shapes={'x': {1: dim1_x}})
def scalar_output(x):
    """
    Returning scalar values from the graph is supported, in addition to Tensor
    outputs. Symbolic shapes are captured and rank is specialized.
    """
    return x.shape[1] + 1