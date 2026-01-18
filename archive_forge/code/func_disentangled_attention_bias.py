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
def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
    if relative_pos is None:
        q = query_layer.size(-2)
        relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions, device=query_layer.device)
    if relative_pos.dim() == 2:
        relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
    elif relative_pos.dim() == 3:
        relative_pos = relative_pos.unsqueeze(1)
    elif relative_pos.dim() != 4:
        raise ValueError(f'Relative position ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
    att_span = self.pos_ebd_size
    relative_pos = relative_pos.long().to(query_layer.device)
    rel_embeddings = rel_embeddings[0:att_span * 2, :].unsqueeze(0)
    if self.share_att_key:
        pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
    else:
        if 'c2p' in self.pos_att_type:
            pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
        if 'p2c' in self.pos_att_type:
            pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads).repeat(query_layer.size(0) // self.num_attention_heads, 1, 1)
    score = 0
    if 'c2p' in self.pos_att_type:
        scale = torch.sqrt(torch.tensor(pos_key_layer.size(-1), dtype=torch.float) * scale_factor)
        c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
        c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]))
        score += c2p_att / scale.to(dtype=c2p_att.dtype)
    if 'p2c' in self.pos_att_type:
        scale = torch.sqrt(torch.tensor(pos_query_layer.size(-1), dtype=torch.float) * scale_factor)
        if key_layer.size(-2) != query_layer.size(-2):
            r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size=self.position_buckets, max_position=self.max_relative_positions, device=query_layer.device)
            r_pos = r_pos.unsqueeze(0)
        else:
            r_pos = relative_pos
        p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
        p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
        p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1, -2)
        score += p2c_att / scale.to(dtype=p2c_att.dtype)
    return score