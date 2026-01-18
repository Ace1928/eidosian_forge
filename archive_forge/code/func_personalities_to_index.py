import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def personalities_to_index(self, personalities):
    """
        Map personalities to their index in the personality dictionary.

        :param personalities:
            list of personalities

        :return:
            list of personality ids
        """
    res = []
    for p in personalities:
        if p in self.personality_to_id:
            res.append(self.personality_to_id[p] + 1)
        else:
            res.append(0)
    return res