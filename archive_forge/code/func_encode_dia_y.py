import torch
import torch.nn as nn
from parlai.agents.transformer.modules import TransformerEncoder
def encode_dia_y(self, y_vecs):
    """
        Encodes a tensor of vectorized candidates.

        :param y_vecs: a [bs, seq_len] or [bs, num_cands, seq_len](?) of vectorized
            candidates
        """
    if y_vecs.dim() == 2:
        y_enc = self.y_dia_head(self.y_dia_encoder(y_vecs))
    elif y_vecs.dim() == 3:
        oldshape = y_vecs.shape
        y_vecs = y_vecs.reshape(oldshape[0] * oldshape[1], oldshape[2])
        y_enc = self.y_dia_head(self.y_dia_encoder(y_vecs))
        y_enc = y_enc.reshape(oldshape[0], oldshape[1], -1)
    return y_enc