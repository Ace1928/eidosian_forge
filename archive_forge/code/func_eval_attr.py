import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def eval_attr(utt, history, attr):
    """
    Given a conversational history and an utterance, compute the requested sentence-
    level attribute for utt.

    Inputs:
        utt: string. The utterance, tokenized and lowercase
        history: a ConvAI2History. This represents the conversation history.
        attr: string. The name of the sentence-level attribute.
    Returns:
        value: float. The value of the attribute for utt.
    """
    assert utt == utt.lower()
    for thing in [history.persona_lines, history.partner_utts, history.own_utts]:
        for line in thing:
            if line != '__SILENCE__':
                assert line == line.lower()
    return ATTR2SENTSCOREFN[attr]((utt, history))