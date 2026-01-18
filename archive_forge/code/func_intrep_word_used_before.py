import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def intrep_word_used_before(dict, hypothesis, history, wt, feat, remove_stopwords):
    """
    Weighted decoding feature function. See explanation above. This feature is 1 for
    words that have already appeared within the hypothesis, 0 otherwise.

    Additional inputs:
      remove_stopwords: bool. If True, stopwords are not included when identifying words
        that have already appeared.
    """
    if hypothesis is not None:
        if remove_stopwords:
            hypothesis = [idx for idx in hypothesis if dict[idx] not in STOPWORDS]
        if len(hypothesis) > 0:
            feat[hypothesis] += wt
    return feat