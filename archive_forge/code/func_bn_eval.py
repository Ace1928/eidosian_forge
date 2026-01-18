import torch
import torch.nn.functional as F
def bn_eval(x: torch.Tensor, bn_weight: torch.Tensor, bn_bias: torch.Tensor, bn_running_mean: torch.Tensor, bn_running_var: torch.Tensor):
    return F.batch_norm(x, bn_running_mean, bn_running_var, bn_weight, bn_bias, training=False)