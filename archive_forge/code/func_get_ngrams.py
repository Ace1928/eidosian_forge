import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def get_ngrams(text, n):
    """
    Returns all ngrams that are in the text.

    Inputs:
        text: string
        n: int
    Returns:
        list of strings (each is a ngram)
    """
    tokens = text.split()
    return [' '.join(tokens[i:i + n]) for i in range(len(tokens) - (n - 1))]