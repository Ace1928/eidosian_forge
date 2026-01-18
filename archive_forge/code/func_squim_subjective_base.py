from typing import Tuple
import torch
import torch.nn as nn
import torchaudio
def squim_subjective_base() -> SquimSubjective:
    """Build :class:`torchaudio.prototype.models.SquimSubjective` model with default arguments."""
    return squim_subjective_model(ssl_type='wav2vec2_base', feat_dim=768, proj_dim=32, att_dim=5)