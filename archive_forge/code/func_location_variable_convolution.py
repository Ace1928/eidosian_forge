from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import ModelOutput, PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_univnet import UnivNetConfig
def location_variable_convolution(self, hidden_states: torch.FloatTensor, kernel: torch.FloatTensor, bias: torch.FloatTensor, dilation: int=1, hop_size: int=256):
    """
        Performs location-variable convolution operation on the input sequence (hidden_states) using the local
        convolution kernel. This was introduced in [LVCNet: Efficient Condition-Dependent Modeling Network for Waveform
        Generation](https://arxiv.org/abs/2102.10815) by Zhen Zheng, Jianzong Wang, Ning Cheng, and Jing Xiao.

        Time: 414 μs ± 309 ns per loop (mean ± std. dev. of 7 runs, 1000 loops each), test on NVIDIA V100.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, in_channels, in_length)`):
                The input sequence of shape (batch, in_channels, in_length).
            kernel (`torch.FloatTensor` of shape `(batch_size, in_channels, out_channels, kernel_size, kernel_length)`):
                The local convolution kernel of shape (batch, in_channels, out_channels, kernel_size, kernel_length).
            bias (`torch.FloatTensor` of shape `(batch_size, out_channels, kernel_length)`):
                The bias for the local convolution of shape (batch, out_channels, kernel_length).
            dilation (`int`, *optional*, defaults to 1):
                The dilation of convolution.
            hop_size (`int`, *optional*, defaults to 256):
                The hop_size of the conditioning sequence.
        Returns:
            `torch.FloatTensor`: the output sequence after performing local convolution with shape (batch_size,
            out_channels, in_length).
        """
    batch, _, in_length = hidden_states.shape
    batch, _, out_channels, kernel_size, kernel_length = kernel.shape
    if in_length != kernel_length * hop_size:
        raise ValueError(f'Dim 2 of `hidden_states` should be {kernel_length * hop_size}) but got {in_length}. Please check `hidden_states` or `kernel` and `hop_size` to make sure they are correct.')
    padding = dilation * int((kernel_size - 1) / 2)
    hidden_states = nn.functional.pad(hidden_states, (padding, padding), 'constant', 0)
    hidden_states = hidden_states.unfold(2, hop_size + 2 * padding, hop_size)
    if hop_size < dilation:
        hidden_states = nn.functional.pad(hidden_states, (0, dilation), 'constant', 0)
    hidden_states = hidden_states.unfold(3, dilation, dilation)
    hidden_states = hidden_states[:, :, :, :, :hop_size]
    hidden_states = hidden_states.transpose(3, 4)
    hidden_states = hidden_states.unfold(4, kernel_size, 1)
    output_hidden_states = torch.einsum('bildsk,biokl->bolsd', hidden_states, kernel)
    output_hidden_states = output_hidden_states.to(memory_format=torch.channels_last_3d)
    bias = bias.unsqueeze(-1).unsqueeze(-1).to(memory_format=torch.channels_last_3d)
    output_hidden_states = output_hidden_states + bias
    output_hidden_states = output_hidden_states.contiguous().view(batch, out_channels, -1)
    return output_hidden_states