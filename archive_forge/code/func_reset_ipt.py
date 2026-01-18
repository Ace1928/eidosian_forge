import warnings
from typing import Any, List, Optional
import torch
from torch import nn
from peft.tuners.lora import LoraLayer
from peft.tuners.tuners_utils import check_adapters_to_merge
from peft.utils import transpose
def reset_ipt(self):
    self.ipt = {}
    self.exp_avg_ipt = {}
    self.exp_avg_unc = {}