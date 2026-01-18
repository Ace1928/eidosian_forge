import torch
import numpy as np
from projects.controllable_dialogue.tasks.build import build
from .stopwords import STOPWORDS
from .nidf import load_word2nidf
from .arora import SentenceEmbedder, load_arora
def bucket_question(ex, ctrl, num_buckets):
    """
    Given an example (where the target response may or may not be a question) and its
    history, probabilistically determine what question-asking CT bucket to use.

    Inputs:
      ex: message dictionary containing a bool field 'question'
      ctrl: string. The name of the CT control. Should be 'question'.
      num_buckets: int. The number of question-asking CT buckets. Assumed to be 11.
    Returns:
      out: int. bucket number.
    """
    assert num_buckets == 11
    is_qn = int(ex['question'])
    assert is_qn in [0, 1]
    is_qn = bool(is_qn)
    if is_qn:
        out = np.random.choice(range(num_buckets), 1, p=PROB_BUCKET_GIVEN_QN)
    else:
        out = np.random.choice(range(num_buckets), 1, p=PROB_BUCKET_GIVEN_NOTQN)
    out = int(out[0])
    return out