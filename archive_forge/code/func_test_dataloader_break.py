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
def test_dataloader_break():
    accelerator = Accelerator()
    first_dset = RegressionDataset(length=80)
    first_dataloader = DataLoader(first_dset, batch_size=16)
    second_dset = RegressionDataset(length=96)
    second_dataloader = DataLoader(second_dset, batch_size=16)
    first_dataloader, second_dataloader = accelerator.prepare(first_dataloader, second_dataloader)
    assert accelerator.gradient_state.active_dataloader is None
    for iteration, _ in enumerate(first_dataloader):
        assert id(accelerator.gradient_state.active_dataloader) == id(first_dataloader)
        if iteration < len(first_dataloader) - 1:
            assert not accelerator.gradient_state.end_of_dataloader
            if iteration == 1:
                for batch_num, _ in enumerate(second_dataloader):
                    assert id(accelerator.gradient_state.active_dataloader) == id(second_dataloader)
                    if batch_num < len(second_dataloader) - 1:
                        assert not accelerator.gradient_state.end_of_dataloader
                    else:
                        assert accelerator.gradient_state.end_of_dataloader
        else:
            assert accelerator.gradient_state.end_of_dataloader
    assert accelerator.gradient_state.active_dataloader is None