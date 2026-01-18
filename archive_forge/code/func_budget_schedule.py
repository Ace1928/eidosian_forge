import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
def budget_schedule(self, step: int):
    tinit = self.peft_config.tinit
    tfinal = self.peft_config.tfinal
    total_step = self.peft_config.total_step
    if step <= tinit:
        budget = self.init_bgt
        mask_ind = False
    elif step > total_step - tfinal:
        budget = self.target_bgt
        mask_ind = True
    else:
        mul_coeff = 1 - (step - tinit) / (total_step - tfinal - tinit)
        budget = int((self.init_bgt - self.target_bgt) * mul_coeff ** 3 + self.target_bgt)
        mask_ind = True if step % self.peft_config.deltaT == 0 else False
    return (budget, mask_ind)