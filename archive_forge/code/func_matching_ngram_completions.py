import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def matching_ngram_completions(comparison_seq, hypothesis, n):
    """
    Return the list of words that if appended to hypothesis, would create a n-gram that
    already exists in comparison_seq. For efficiency, this function represents words as
    integers not strings.

    Inputs:
        comparison_seq: list of integers
        hypothesis: list of integers or None
        n: integer

    Output:
        bad_words: list of integers
    """
    if hypothesis is None or len(hypothesis) < n - 1 or len(comparison_seq) < n:
        return []
    hypothesis = [int(i) for i in hypothesis]
    comparison_seq = [int(i) for i in comparison_seq]
    n_minus_1_gram = hypothesis[-(n - 1):]
    bad_words = [comparison_seq[i] for i in range(n - 1, len(comparison_seq)) if comparison_seq[i - (n - 1):i] == n_minus_1_gram]
    return bad_words