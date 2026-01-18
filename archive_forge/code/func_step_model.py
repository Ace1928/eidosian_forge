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
def step_model(model, input, target, accelerator, do_backward=True):
    model.train()
    output = model(input)
    loss = F.mse_loss(output, target.to(output.device))
    if not do_backward:
        loss /= accelerator.gradient_accumulation_steps
        loss.backward()
    else:
        accelerator.backward(loss)