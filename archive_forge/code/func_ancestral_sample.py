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
@add_start_docstrings('\n        Generates music tokens based on the provided `labels. Will start at the desired prior level and automatically\n        upsample the sequence. If you want to create the audio, you should call `model.decode(tokens)`, which will use\n        the VQ-VAE decoder to convert the music tokens to raw audio.\n\n        Args:\n            labels (`List[torch.LongTensor]`) :\n                List of length `n_sample`, and shape `(self.levels, 4 + self.config.max_nb_genre +\n                lyric_sequence_length)` metadata such as `artist_id`, `genre_id` and the full list of lyric tokens\n                which are used to condition the generation.\n            n_samples (`int`, *optional*, default to 1) :\n                Number of samples to be generated in parallel.\n        ')
def ancestral_sample(self, labels, n_samples=1, **sampling_kwargs) -> List[torch.LongTensor]:
    """
        Example:

        ```python
        >>> from transformers import AutoTokenizer, JukeboxModel, set_seed

        >>> model = JukeboxModel.from_pretrained("openai/jukebox-1b-lyrics", min_duration=0).eval()
        >>> tokenizer = AutoTokenizer.from_pretrained("openai/jukebox-1b-lyrics")

        >>> lyrics = "Hey, are you awake? Can you talk to me?"
        >>> artist = "Zac Brown Band"
        >>> genre = "Country"
        >>> metas = tokenizer(artist=artist, genres=genre, lyrics=lyrics)
        >>> set_seed(0)
        >>> music_tokens = model.ancestral_sample(metas.input_ids, sample_length=400)

        >>> with torch.no_grad():
        ...     model.decode(music_tokens)[:, :10].squeeze(-1)
        tensor([[-0.0219, -0.0679, -0.1050, -0.1203, -0.1271, -0.0936, -0.0396, -0.0405,
            -0.0818, -0.0697]])
        ```
        """
    sample_levels = sampling_kwargs.pop('sample_levels', list(range(len(self.priors))))
    music_tokens = [torch.zeros(n_samples, 0, dtype=torch.long, device=labels[0].device) for _ in range(len(self.priors))]
    music_tokens = self._sample(music_tokens, labels, sample_levels, **sampling_kwargs)
    return music_tokens