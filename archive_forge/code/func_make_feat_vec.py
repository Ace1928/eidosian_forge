import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def make_feat_vec(self, dict):
    """
        Construct the NIDF feature vector for the given dict.
        """
    print('Constructing NIDF feature vector...')
    self.NIDF_FEATS = torch.zeros(len(dict))
    num_oovs = 0
    for idx in range(len(dict)):
        word = dict[idx]
        if word in word2nidf:
            if word[0] == '@' and word[-1] == '@':
                continue
            nidf = word2nidf[word]
            self.NIDF_FEATS[idx] = nidf
        else:
            num_oovs += 1
    print('Done constructing NIDF feature vector; of %i words in dict there were %i words with unknown NIDF; they were marked as NIDF=0.' % (len(dict), num_oovs))