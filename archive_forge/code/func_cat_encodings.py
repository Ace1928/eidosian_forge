import torch
from torch import nn
from parlai.agents.transformer.modules import (
from projects.personality_captions.transresnet.modules import (
def cat_encodings(self, tensors):
    """
        Concatenate non-`None` encodings.

        :param tensors:
            list tensors to concatenate

        :return:
            concatenated tensors
        """
    tensors = [t for t in tensors if t is not None]
    return torch.cat([t.unsqueeze(1) for t in tensors], dim=1)