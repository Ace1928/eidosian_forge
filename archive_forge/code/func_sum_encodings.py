import torch
from torch import nn
from torch import optim
from parlai.agents.transformer.modules import TransformerEncoder
from parlai.agents.transformer import transformer as Transformer
def sum_encodings(self, addends):
    """
        Add up a list of encodings, some of which may be `None`.

        :param addends:
            tensors to add

        :return:
            sum of non-`None` addends
        """
    addends = [a for a in addends if a is not None]
    return sum(addends) if len(addends) > 0 else None