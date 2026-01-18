import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def sample_single_window(self, music_tokens, labels, offset, sampling_kwargs, level, start, max_batch_size):
    prior = self.priors[level]
    n_samples = music_tokens[0].shape[0]
    n_ctx = prior.n_ctx
    end = start + n_ctx
    previous_sampled_tokens = music_tokens[level][:, start:end]
    sample_tokens = sampling_kwargs.get('sample_tokens', None)
    if 'sample_tokens' in sampling_kwargs:
        sample_tokens = end - start
    conditioning_tokens = previous_sampled_tokens.shape[1]
    new_tokens = sample_tokens - previous_sampled_tokens.shape[1]
    logger.info(f'Sampling {sample_tokens} tokens for [{start},{start + sample_tokens}]. Conditioning on {conditioning_tokens} tokens')
    if new_tokens <= 0:
        return music_tokens
    music_tokens_conds = prior.get_music_tokens_conds(music_tokens, start, end)
    metadata = prior.get_metadata(labels, start, self.total_length, offset)
    music_tokens_list = self.split_batch(previous_sampled_tokens, n_samples, max_batch_size)
    music_tokens_conds_list = self.split_batch(music_tokens_conds, n_samples, max_batch_size)
    metadata_list = self.split_batch(metadata, n_samples, max_batch_size)
    tokens = []
    iterator = tqdm(zip(music_tokens_list, music_tokens_conds_list, metadata_list), leave=False)
    for music_tokens_i, music_tokens_conds_i, metadata_i in iterator:
        name = ['Ancestral', 'Primed'][music_tokens_i.shape[1] == 0]
        iterator.set_description(f'[prior level {level}] {name} Sampling {sample_tokens} tokens out of {self.total_length // prior.raw_to_tokens}', refresh=True)
        tokens_i = prior.sample(n_samples=music_tokens_i.shape[0], music_tokens=music_tokens_i, music_tokens_conds=music_tokens_conds_i, metadata=metadata_i, **sampling_kwargs)
        tokens.append(tokens_i)
    sampled_tokens = torch.cat(tokens, dim=0)
    music_tokens_new = sampled_tokens[:, -new_tokens:]
    music_tokens[level] = torch.cat([music_tokens[level], music_tokens_new], dim=1)
    return music_tokens