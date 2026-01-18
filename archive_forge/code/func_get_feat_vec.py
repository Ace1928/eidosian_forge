import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def get_feat_vec(self, dict):
    """
        Return the NIDF feature vector.

        If necessary, construct it first.
        """
    if self.NIDF_FEATS is None:
        self.make_feat_vec(dict)
    return self.NIDF_FEATS