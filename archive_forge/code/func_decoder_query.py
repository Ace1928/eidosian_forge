import abc
import math
from dataclasses import dataclass
from functools import reduce
from operator import __add__
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, meshgrid, prune_linear_layer
from ...utils import (
from .configuration_perceiver import PerceiverConfig
def decoder_query(self, inputs, modality_sizes, inputs_without_pos=None, subsampled_points=None):
    inputs = restructure(modality_sizes, inputs)
    subsampled_points = subsampled_points or {}
    decoder_queries = {}
    for modality, decoder in self.modalities.items():
        input_without_pos = None
        if inputs_without_pos is not None:
            input_without_pos = inputs_without_pos.get(modality, None)
        query = decoder.decoder_query(inputs=inputs[modality], modality_sizes=None, inputs_without_pos=input_without_pos, subsampled_points=subsampled_points.get(modality, None))
        decoder_queries[modality] = query

    def embed(modality, x):
        x = torch.reshape(x, [x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])
        pos = self.padding[modality]
        pos = torch.broadcast_to(pos, [x.shape[0], x.shape[1], self.num_query_channels - x.shape[2]])
        return torch.cat([x, pos], dim=2)
    return torch.cat([embed(modality, decoder_queries[modality]) for modality in sorted(self.modalities.keys())], dim=1)