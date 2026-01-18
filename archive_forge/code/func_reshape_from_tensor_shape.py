import torch
import torch.onnx
def reshape_from_tensor_shape(x, shape):
    return torch._reshape_from_tensor(x, shape)