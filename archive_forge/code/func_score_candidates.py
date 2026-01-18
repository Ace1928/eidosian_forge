from functools import lru_cache
import torch
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .modules import MemNN, opt_to_kwargs
def score_candidates(self, batch, cand_vecs, cand_encs=None):
    mems = self._build_mems(batch.memory_vecs)
    pad_mask = None
    if mems is not None:
        pad_mask = (mems != self.NULL_IDX).sum(dim=-1) == 0
    if cand_encs is not None:
        state, _ = self.model(batch.text_vec, mems, None, pad_mask)
    else:
        state, cand_encs = self.model(batch.text_vec, mems, cand_vecs, pad_mask)
    scores = self._score(state, cand_encs)
    return scores