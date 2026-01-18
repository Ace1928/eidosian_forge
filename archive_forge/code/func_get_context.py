from collections.abc import Sequence
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, LayerNorm, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import (
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_deberta_v2 import DebertaV2Config
def get_context(self):
    if self.context_stack is not None:
        if self.count >= len(self.context_stack):
            self.context_stack.append(DropoutContext())
        ctx = self.context_stack[self.count]
        ctx.dropout = self.drop_prob
        self.count += 1
        return ctx
    else:
        return self.drop_prob