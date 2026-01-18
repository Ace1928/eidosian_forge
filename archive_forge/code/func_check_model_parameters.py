from copy import deepcopy
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from accelerate.accelerator import Accelerator, GradientAccumulationPlugin
from accelerate.state import GradientState
from accelerate.test_utils import RegressionDataset, RegressionModel
from accelerate.utils import DistributedType, set_seed
def check_model_parameters(model_a, model_b, did_step, iteration, **kwargs):
    for param, grad_param in zip(model_a.parameters(), model_b.parameters()):
        if not param.requires_grad:
            continue
        if not did_step:
            assert torch.allclose(param.grad, grad_param.grad, **kwargs) is False, f'Gradients in sync when they should not be at iteration {iteration}:\nmodel_a grad ({param.grad}) == model_b grad ({grad_param.grad})'
        else:
            assert torch.allclose(param.grad, grad_param.grad, **kwargs) is True, f'Gradients not in sync when they should be at iteration {iteration}:\nmodel_a grad ({param.grad}) != model_b grad ({grad_param.grad})'