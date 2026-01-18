import torch
from torch import nn
from parlai.agents.transformer.modules import (
from projects.personality_captions.transresnet.modules import (
def get_rep(self, encodings, batchsize=None):
    """
        Get the multimodal representation of the encodings.

        :param encodings:
            list of encodings
        :param batchsize:
            size of batch

        :return:
            final multimodal representations
        """
    if not self.multimodal:
        rep = self.sum_encodings(encodings)
    else:
        if self.multimodal_combo == 'sum':
            encodings = self.sum_encodings(encodings).unsqueeze(1)
        elif self.multimodal_combo == 'concat':
            encodings = self.cat_encodings(encodings)
        all_one_mask = torch.ones(encodings.size()[:2])
        if self.use_cuda:
            all_one_mask = all_one_mask.cuda()
        rep = self.multimodal_encoder(encodings, all_one_mask)
    if rep is None:
        rep = torch.stack([self.blank_encoding for _ in range(batchsize)])
    return rep