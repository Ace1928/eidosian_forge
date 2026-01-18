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
def get_training_setup(accelerator, sched=False):
    """Returns everything needed to perform basic training"""
    set_seed(42)
    model = RegressionModel()
    ddp_model = deepcopy(model)
    dset = RegressionDataset(length=80)
    dataloader = DataLoader(dset, batch_size=16)
    model.to(accelerator.device)
    if sched:
        opt = AdamW(params=model.parameters(), lr=0.001)
        ddp_opt = AdamW(params=ddp_model.parameters(), lr=0.001)
        sched = LambdaLR(opt, lr_lambda=lambda epoch: epoch ** 0.65)
        ddp_sched = LambdaLR(ddp_opt, lr_lambda=lambda epoch: epoch ** 0.65)
    if sched:
        ddp_model, ddp_opt, ddp_sched, dataloader = accelerator.prepare(ddp_model, ddp_opt, ddp_sched, dataloader)
    else:
        ddp_model, dataloader = accelerator.prepare(ddp_model, dataloader)
    if sched:
        return (model, opt, sched, dataloader, ddp_model, ddp_opt, ddp_sched)
    return (model, ddp_model, dataloader)