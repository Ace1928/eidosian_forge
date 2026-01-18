import torch
import torch.nn as nn
from parlai.utils.torch import neginf
from functools import lru_cache
@staticmethod
@lru_cache(maxsize=128)
def position_matrix(J, d, use_cuda):
    """
        Build matrix of position encoding coeffiencents.

        See https://papers.nips.cc/paper/5846-end-to-end-memory-networks,
        section 4.1 Model Details: Sentence Representation.

        :param J:
            number of words in the sequence

        :param d:
            dimension of the embedding

        :returns:
            Position Encoding matrix
        """
    m = torch.Tensor(J, d)
    for k in range(1, d + 1):
        for j in range(1, J + 1):
            m[j - 1, k - 1] = 1 - j / J - k / d * (1 - 2 * j / J)
    if use_cuda:
        m = m.cuda()
    return m