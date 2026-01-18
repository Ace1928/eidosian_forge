import math
import os
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, SmoothL1Loss
from ...activations import ACT2FN, gelu
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_lxmert import LxmertConfig
def output_fc(self, lang_input, visual_input):
    lang_inter_output = self.lang_inter(lang_input)
    visual_inter_output = self.visn_inter(visual_input)
    lang_output = self.lang_output(lang_inter_output, lang_input)
    visual_output = self.visn_output(visual_inter_output, visual_input)
    return (lang_output, visual_output)