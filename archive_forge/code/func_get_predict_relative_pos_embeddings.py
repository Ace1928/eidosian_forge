import copy
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import Tensor, nn
from torch.nn import LayerNorm
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_xlm_prophetnet import XLMProphetNetConfig
def get_predict_relative_pos_embeddings(self, hidden_states, attn_weights, position_ids, predict_relative_position_buckets):
    batch_size, sequence_length = hidden_states.shape[0:2]
    if predict_relative_position_buckets is None:
        key_sequence_length = attn_weights.shape[-1]
        assert position_ids[0][0] == key_sequence_length - 1, '`position_ids` are incorrect. They should be of the format 1 2 3 4 5 ... (key_sequence_length - 1)'
        relative_positions = torch.arange(0, key_sequence_length).unsqueeze(0).unsqueeze(0).repeat(batch_size, sequence_length, 1).to(position_ids.device)
        relative_positions = relative_positions - position_ids.unsqueeze(0).repeat(batch_size, sequence_length, 1)
        predict_relative_position_buckets = compute_relative_buckets(self.num_buckets, self.relative_max_distance, relative_positions, False)
    hidden_states = hidden_states.transpose(1, 2)
    rel_pos_embeddings = self.relative_pos_embeddings(hidden_states)
    rel_pos_embeddings = rel_pos_embeddings.view(hidden_states.shape[:-1] + (self.num_buckets, self.num_attn_heads))
    rel_pos_embeddings = rel_pos_embeddings.permute(0, 2, 1, 4, 3)
    rel_pos_embeddings = rel_pos_embeddings.reshape(-1, self.num_buckets)
    predict_relative_position_buckets = predict_relative_position_buckets.unsqueeze(0)
    predict_relative_position_buckets = predict_relative_position_buckets.repeat(self.ngram, 1, self.num_attn_heads, 1)
    predict_relative_position_buckets = predict_relative_position_buckets.view(-1, predict_relative_position_buckets.size(-1)).long()
    predict_relative_pos_embeddings = torch.gather(rel_pos_embeddings, dim=1, index=predict_relative_position_buckets)
    predict_relative_pos_embeddings = predict_relative_pos_embeddings.view(batch_size, self.ngram, self.num_attn_heads, sequence_length, -1)
    return predict_relative_pos_embeddings