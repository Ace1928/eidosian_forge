import torch
from torch import nn
from parlai.agents.transformer.modules import (
from projects.personality_captions.transresnet.modules import (
def choose_topk(self, idx, encoded, candidates, candidates_encoded, one_cand_set, k):
    """
        Choose top k best responses for a single example.

        :param idx:
            idx of example in encoded
        :param encoded:
            full matrix of encoded representations (for the whole batch)
        :param candidates:
            list of candidates
        :param candidates_encoded:
            encoding of the candidates
        :param one_cand_set:
            true if there is one set of candidates for each example
        :param k:
            how many ranked responses to return

        :return:
            ranked list of k responses
        """
    encoding = encoded[idx:idx + 1, :]
    scores = torch.mm(candidates_encoded[idx] if not one_cand_set else candidates_encoded, encoding.transpose(0, 1))
    if k >= 1:
        _, index_top = torch.topk(scores, k, dim=0)
    else:
        _, index_top = torch.topk(scores, scores.size(0), dim=0)
    return [candidates[idx][idx2] if not one_cand_set else candidates[idx2] for idx2 in index_top.unsqueeze(1)]