import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def nidf(dict, hypothesis, history, wt, feat):
    """
    Weighted decoding feature function.

    See explanation above. This feature is equal to the NIDF (normalized inverse
    document frequency) score for each word. The score is always between 0 and 1.
    """
    feat += wt * nidf_feats.get_feat_vec(dict)
    return feat