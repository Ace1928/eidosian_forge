from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.core.build_data import modelzoo_path
import torchtext.vocab as vocab
from parlai.utils.misc import TimeLogger
from collections import Counter, deque
import numpy as np
import os
import pickle
import torch
def embed_sent(self, sent, rem_first_sv=True):
    """
        Produce a Arora-style sentence embedding for a given sentence.

        Inputs:
          sent: tokenized sentence; a list of strings
          rem_first_sv: If True, remove the first singular value when you compute the
            sentence embddings. Otherwise, don't remove it.
        Returns:
          sent_emb: tensor length glove_dim, or None.
              If sent_emb is None, that's because all of the words were OOV for GloVe.
        """
    if self.tt_embs is None:
        self.get_glove_embs()
    tokens = [t for t in sent if t in self.tt_embs.stoi]
    if len(tokens) == 0:
        print('WARNING: tried to embed utterance %s but all tokens are OOV for GloVe. Returning embedding=None' % sent)
        return None
    word_embs = [self.tt_embs.vectors[self.tt_embs.stoi[t]] for t in tokens]
    unigram_probs = [self.word2prob[t] if t in self.word2prob else self.min_word_prob for t in tokens]
    smooth_inverse_freqs = [self.arora_a / (self.arora_a + p) for p in unigram_probs]
    sent_emb = sum([word_emb * wt for word_emb, wt in zip(word_embs, smooth_inverse_freqs)]) / len(word_embs)
    if rem_first_sv:
        sent_emb = remove_first_sv(sent_emb, self.first_sv)
    return sent_emb