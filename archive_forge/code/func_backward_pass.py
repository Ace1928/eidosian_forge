import sys
from collections import namedtuple
from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
from torch import nn
from torch.autograd.function import Function
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import CausalLMOutput, MaskedLMOutput, QuestionAnsweringModelOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import (
from .configuration_reformer import ReformerConfig
def backward_pass(self, next_attn_output, hidden_states, grad_attn_output, grad_hidden_states, attention_mask=None, head_mask=None, buckets=None):
    assert self.training, 'If you want to train `ReformerModel` and its variations, make sure to use `model.train()` to put the model into training mode.'
    with torch.enable_grad():
        next_attn_output.requires_grad = True
        torch.manual_seed(self.feed_forward_seed)
        res_hidden_states = self.feed_forward(next_attn_output)
        res_hidden_states.backward(grad_hidden_states, retain_graph=True)
    with torch.no_grad():
        hidden_states = hidden_states - res_hidden_states
        del res_hidden_states
        grad_attn_output = grad_attn_output + next_attn_output.grad
        next_attn_output.grad = None
    with torch.enable_grad():
        hidden_states.requires_grad = True
        torch.manual_seed(self.attention_seed)
        output = self.attention(hidden_states=hidden_states, head_mask=head_mask, attention_mask=attention_mask, buckets=buckets).hidden_states
        output.backward(grad_attn_output, retain_graph=True)
    with torch.no_grad():
        attn_output = next_attn_output - output
        del output, next_attn_output
        grad_hidden_states = grad_hidden_states + hidden_states.grad
        hidden_states.grad = None
        hidden_states = hidden_states.detach()
    return ReformerBackwardOutput(attn_output=attn_output, hidden_states=hidden_states, grad_attn_output=grad_attn_output, grad_hidden_states=grad_hidden_states)