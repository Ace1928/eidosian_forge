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
def get_emb_matrix(self, dictionary):
    """
        Construct an embedding matrix containing pretrained GloVe vectors for all words
        in dictionary, and store in self.emb_matrix. This is needed for response-
        relatedness weighted decoding.

        Inputs:
          dictionary: ParlAI dictionary
        """
    print('Constructing GloVe emb matrix for response-relatedness weighted decoding...')
    self.emb_matrix = []
    oov_indices = []
    for idx in range(len(dictionary)):
        word = dictionary[idx]
        if word in self.tt_embs.stoi:
            word_emb = self.tt_embs.vectors[self.tt_embs.stoi[word]]
        else:
            word_emb = torch.zeros(self.glove_dim)
            oov_indices.append(idx)
        self.emb_matrix.append(word_emb)
    self.emb_matrix = np.stack(self.emb_matrix)
    print('Done constructing GloVe emb matrix; found %i OOVs of %i words' % (len(oov_indices), len(dictionary)))
    self.emb_matrix_norm = np.linalg.norm(self.emb_matrix, axis=1)
    for idx in oov_indices:
        self.emb_matrix_norm[idx] = 1.0