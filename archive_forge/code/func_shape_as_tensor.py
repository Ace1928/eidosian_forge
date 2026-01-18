import torch
import torch.onnx
def shape_as_tensor(x):
    return torch._shape_as_tensor(x)