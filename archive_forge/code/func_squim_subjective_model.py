from typing import Tuple
import torch
import torch.nn as nn
import torchaudio
def squim_subjective_model(ssl_type: str, feat_dim: int, proj_dim: int, att_dim: int) -> SquimSubjective:
    """Build a custome :class:`torchaudio.prototype.models.SquimSubjective` model.

    Args:
        ssl_type (str): Type of self-supervised learning (SSL) models.
            Must be one of ["wav2vec2_base", "wav2vec2_large"].
        feat_dim (int): Feature dimension of the SSL feature representation.
        proj_dim (int): Output dimension of projection layer.
        att_dim (int): Dimension of attention scores.
    """
    ssl_model = getattr(torchaudio.models, ssl_type)()
    projector = nn.Linear(feat_dim, proj_dim)
    predictor = Predictor(proj_dim * 2, att_dim)
    return SquimSubjective(ssl_model, projector, predictor)