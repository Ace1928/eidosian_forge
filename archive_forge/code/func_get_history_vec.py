from parlai.agents.bert_ranker.bert_dictionary import BertDictionaryAgent
from parlai.agents.bert_ranker.helpers import (
from parlai.core.torch_agent import History
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.utils.misc import warn_once
from parlai.zoo.bert.build import download
from collections import deque
import os
import torch
def get_history_vec(self):
    """
        Override from parent class to possibly add [SEP] token.
        """
    if not self.sep_last_utt or len(self.history_vecs) <= 1:
        return super().get_history_vec()
    history = deque(maxlen=self.max_len)
    for vec in self.history_vecs[:-1]:
        history.extend(vec)
        history.extend(self.delimiter_tok)
    history.extend([self.dict.end_idx])
    history.extend(self.history_vecs[-1])
    return history