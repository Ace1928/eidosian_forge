import dropout_layer_norm
import torch
from torch.nn import init
def layer_norm(x, weight, bias, epsilon):
    return DropoutAddLayerNormFn.apply(x, None, weight, bias, None, None, 0.0, epsilon, False)