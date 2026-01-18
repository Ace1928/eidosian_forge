import torch
import torch.ao.nn.quantized.dynamic as nnqd
import torch.ao.nn.intrinsic as nni
@classmethod
def from_reference(cls, ref_qlinear_relu):
    return super().from_reference(ref_qlinear_relu[0])