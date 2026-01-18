import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def forward_personality(self, personalities, personalities_tensor):
    """
        Encode personalities.

        :param personalities:
            list of personalities, one per example
        :param personalities_tensor:
            (optional) list of personality representations, usually a one-hot
            vector if specified

        :return:
            encoded representation of the personalities
        """
    pers_encoded = None
    if personalities is not None:
        if personalities_tensor is not None:
            pers_feature = personalities_tensor
        else:
            res = torch.FloatTensor(len(personalities), self.num_personalities).fill_(0)
            p_to_i = self.personalities_to_index(personalities)
            for i, index in enumerate(p_to_i):
                res[i, index] = 1
            if self.use_cuda:
                res = res.cuda()
            pers_feature = res
        pers_encoded = self.personality_encoder(pers_feature)
    return pers_encoded