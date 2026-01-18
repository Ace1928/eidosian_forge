from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def prune_lstm_output_linear(lstm: nn.LSTM, getitem: Callable, linear: nn.Linear) -> None:
    prune_lstm_output_layernorm_linear(lstm, getitem, None, linear)