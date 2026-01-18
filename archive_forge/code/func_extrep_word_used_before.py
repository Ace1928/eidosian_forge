import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def extrep_word_used_before(dict, hypothesis, history, wt, feat, remove_stopwords, person):
    """
    Weighted decoding feature function. See explanation above. This feature is 1 for
    words that have already been used earlier in the conversation; otherwise 0.

    Additional inputs:
      remove_stopwords: bool. If True, stopwords are not included when identifying words
        that have already appeared.
      person: If 'self', identify words that have already been used by self (bot).
        If 'partner', identify words that have already been used by partner (human).
    """
    if person == 'self':
        prev_utts = history.own_utts
    elif person == 'partner':
        prev_utts = history.partner_utts
    else:
        raise ValueError("person must be 'self' or 'partner', but it is: ", person)
    if len(prev_utts) == 0:
        return feat
    prev_words = [dict.txt2vec(utt) for utt in prev_utts]
    prev_words = list(set(flatten(prev_words)))
    if remove_stopwords:
        prev_words = [idx for idx in prev_words if dict[idx] not in STOPWORDS]
    if len(prev_words) > 0:
        feat[prev_words] += wt
    return feat