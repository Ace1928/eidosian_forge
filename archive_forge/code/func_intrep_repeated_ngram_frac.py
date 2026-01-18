import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def intrep_repeated_ngram_frac(utt, history, n):
    """
    Sentence-level attribute function.

    See explanation above.
    Returns the fraction of n-grams in utt that are repeated.
    Additional inputs:
      n: int, the size of the n-grams considered.
    """
    assert utt.strip() != ''
    ngrams = get_ngrams(utt, n)
    return intrep_frac(ngrams)